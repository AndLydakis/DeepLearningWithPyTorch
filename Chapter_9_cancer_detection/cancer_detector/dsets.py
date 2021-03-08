from collections import namedtuple
import SimpleITK as sitk
import functools
from luna_util.util import XyzTuple, irc2xyz, xyz2irc
from luna_util.logconf import logging
from luna_util.disk import GzipDisk, getCache

import torch
import torch.cuda
from torch.utils.data import Dataset

raw_cache = getCache('part2ch10_raw')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple', 'isNodule_bool, diameter_mm, series_uid, center_xyz'
)


# standard library in-memory caching
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):  # screen out data that have not been loaded yet
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    print('Present on disk: {}'.format(presentOnDisk_set))
    diameter_dict = {}
    # group annotations by UID
    with open('data/part2/luna/annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            # get the dict element by uid, or an empty array as default,
            # and append the current center and diameter
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )
    # build list of candidates
    candidateInfo_list = []
    with open('data/part2/luna/annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    # break if the candidate is too far apart from the annotation
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:  # execute after loop completes normally
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool=isNodule_bool,
                diameter_mm=candidateDiameter_mm,
                series_uid=series_uid,
                center_xyz=candidateCenter_xyz
            ))
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob('data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid))[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)
        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        # convert direction to array and reshape from 9 element array to 3x3
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidates(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a
        )
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_idx = int(round(center_val - width_irc[axis] / 2))
            end_idx = int(start_idx + width_irc[axis])
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_idx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_idx = 0
                end_idx = int(width_irc[axis])

            if end_idx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_idx = self.hu_a.shape[axis]
                start_idx = int(self.hu_a.shape[axis] - width_irc[axis])
            slice_list.append(slice(start_idx, end_idx))
        ct_chunk = self.hu_a[tuple(slice_list)]
        return ct_chunk, center_irc


# We’re  caching  the  getCt  returnvalue in memory so that we can repeatedly ask for the same Ct
# instance without having to  reload  all  of  the  data  from  disk
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


# The getCtRawCandidate function that calls getCtalso has its outputs cached, however
# so  after  our  cache  is  populated,  getCt  won’t  ever  be  called
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_ir


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None
                 ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        log.info('{!r}: {} {} samples'.format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training"
        ))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, idx):
        candidateInfo_tup = self.candidateInfo_list[idx]
        witdh_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)  # add the channel dimension

        # positive tensor is two elements, for two classes that cross entropy expects
        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ], dtype=torch.long)

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc)
        )
