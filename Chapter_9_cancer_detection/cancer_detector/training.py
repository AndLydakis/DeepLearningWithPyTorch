import argparse
import datetime
import os
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from luna_util.util import enumerateWithEstimate
from .dsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from luna_util.logconf import logging
from .model import UNetWrapper, SegmentationAugmentation

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
# METRICS_LABEL_NDX = 0
# METRICS_PRED_NDX = 1
# METRICS_LOSS_NDX = 2
# METRICS_SIZE = 3

METRICS_LOSS_NDX = 1
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9
METRICS_SIZE = 10


class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]  # get arguments from command line if not provided

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help='Number of processes to use for data loading', default=8, type=int)
        parser.add_argument('--batch-size', help='Batch Size', default=32, type=int)
        parser.add_argument('--augmented', help='Augment the training data', action='store_true', default=False)
        parser.add_argument('--augment-flip', help='Augment the training data by flipping', action='store_true',
                            default=False)
        parser.add_argument('--augmented-offset', help='Augment the training data by adding offset',
                            action='store_true', default=False)
        parser.add_argument('--augmented-scale', help='Augment the training data by adding scale', action='store_true',
                            default=False)
        parser.add_argument('--augmented-rotate', help='Augment the training data by adding rotation',
                            action='store_true', default=False)
        parser.add_argument('--augmented-noise', help='Augment the training data by adding noise', action='store_true',
                            default=False)
        parser.add_argument('--epochs', help='Number of training epochs', default=1, type=int)
        parser.add_argument('--tb-prefix', help='Tensorboard prefix', default='luna')
        parser.add_argument('comment', help="Comment suffix for Tensorboard run.", nargs='?', default='dwlpt')

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.totalTrainingSamples_count = 0

        self.trn_writer = None
        self.val_writer = None

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.segmentation_model, self.augmentation_model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        segmentation_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv'
        )
        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)
        return segmentation_model, augmentation_model

    def initOptimizer(self):
        return Adam(self.segmentation_model.parameters())

    def initTrainDl(self):
        train_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        return train_dl

    def initValDl(self):
        val_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_idx in range(1, self.cli_args.epochs + 1):
            print('Epoch {}/{}, {}/{} batches of size {}*{}'.format(epoch_idx, self.cli_args.epochs, len(train_dl),
                                                                    len(val_dl), self.cli_args.batch_size, (
                                                                        torch.cuda.device_count() if self.use_cuda else 1)))
            log.info('Epoch {}/{}, {}/{} batches of size {}*{}'.format(epoch_idx, self.cli_args.epochs, len(train_dl),
                                                                       len(val_dl), self.cli_args.batch_size, (
                                                                           torch.cuda.device_count() if self.use_cuda else 1)))

            trnMetrics_t = self.doTraining(epoch_idx, train_dl)
            self.logMetrics(epoch_idx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_idx, val_dl)
            self.logMetrics(epoch_idx, 'val', trnMetrics_t)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_idx, train_dl):
        self.segmentation_model.train()
        train_dl.dataset.shuffleSamples()

        batch_iter = enumerateWithEstimate(
            train_dl,
            'E{} training'.format(epoch_idx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_idx, val_dl):
        with torch.no_grad():
            self.segmentation_model.eval()

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_idx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.segmentation_model(input_g)

        diceLoss_g = self.diceLoss(prediction_g, label_g)
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)

        start_ndx = batch_ndx * batch_size
        end_dx = start_ndx + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1] > classificationThreshold).to(torch.float32)
            tp = (predictionBool_g * label_g).sum(dim=[1, 2, 3])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1, 2, 3])
            tp = (predictionBool_g * (~label_g)).sum(dim=[1, 2, 3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_dx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_dx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_dx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_dx] = fp

        return diceLoss_g.mean() + fnLoss_g.mean() * 8

    def diceLoss(self, prediction_g, label_g, epsilon=1):
        diceLabel_g = label_g.sum(dim=[1, 2, 3])
        dicePrediction_g = prediction_g.sum(dim=[1, 2, 3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1, 2, 3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g

    def logImages(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval()

        images = sorted(dl.dataset.series_list)[:12]
        for series_ndx, series_uid in enumerate(images):
            ct = getCt(series_uid)

            for slice_ndx in range(6):
                ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5
                sample_tup = dl.dataset.getitem_fullSlice(series_uid, ct_ndx)

                ct_t, label_t, series_uid, ct_ndx = sample_tup

                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = pos_g = label_t.to(self.device).unsqueeze(0)

                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                label_a = label_g.cpu().numpy[0][0] > 0.5

                ct_t[:-1, :, :] /= 2000
                ct_t[:-1, :, :] += 0.5

                ctSlice_a = ct[dl.dataset.contextSlices_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                image_a[:, :, 0] += prediction_a & (1 - label_a)
                image_a[:, :, 0] += (1 - prediction_a) & label_a
                image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5
                image_a[:, :, 1] += prediction_a & label_a
                image_a *= 0.5

                image_a.clip(0, 1, image_a)

                writer = getattr(self, mode_str + '_writer')
                writer.add_image(
                    f'{mode_str}/{series_ndx}_prediction_{slice_ndx}',
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformat='HWC'
                )

                if epoch_ndx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                    image_a[:, :, 1] += label_a

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1

                    writer.add_image(
                        f'{mode_str}/{series_ndx}_prediction_{slice_ndx}',
                        image_a,
                        self.totalTrainingSamples_count,
                        dataformat='HWC'
                    )
            writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        log.info('E{} {}'.format(epoch_ndx, type(self).__name__))
        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)

        assert np.isinfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()
        metrics_dict['percent_all/tp'] = metrics_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = metrics_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = metrics_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] / (
                (sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        precision = metrics_dict['pr/recall'] = sum_a[METRICS_TP_NDX] / (
                (sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info((
            'E{} {:8} {loss/all:.4f} loss, {percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-5.1f}% fp ').format(
            epoch_ndx, mode_str + '_all', **metrics_dict))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()
        score = metrics_dict['pr/recall']

        return score

    def saveModeL(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)
        log.info('Saved model params to {}'.format(file_path))

        if isBest:
            best_path = os.path.join(
                'data-unversioned', 'part2', 'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state'
            )
            shutil.copyfile(file_path, best_path)
            log.info('Saved model params to {}'.format(best_path))

        with open(file_path, 'rb') as f:
            log.info('SHA1: ' + hashlib.sha1(f.read()).hexdigest())


if __name__ == '__main__':
    SegmentationTrainingApp().main()
