import argparse
import sys


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]  # get arguments from command line if not provided

        self.use_cuda = torch.cuda.is_available()
        self.device = roch
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help='Number of processes to use for data loading', default=8, type=int)
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    LunaTrainingApp.main()
