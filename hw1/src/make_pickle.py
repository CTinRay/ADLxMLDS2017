import argparse
import pdb
import pickle
import sys
import traceback
from utils import DataProcessor


def main():
    parser = argparse.ArgumentParser(description='Load data as pickle')
    parser.add_argument('path', type=str, help='Data directory path')
    parser.add_argument('out', type=str, help='Filename of output pickle')
    parser.add_argument('mean_std', type=str,
                        help='Filename of pickle to store mean and std')
    parser.add_argument('--test_only', type=bool, default=False,
                        help='Whether or not only read test data.')
    parser.add_argument('--feature', type=str, default='mfcc',
                        help='mfcc or fbank')
    args = parser.parse_args()

    print('Processing data.', file=sys.stderr)
    dp = DataProcessor(args.path, feature=args.feature)

    print('Start dumping pickle.', file=sys.stderr)
    with open(args.out, 'wb') as f:
        pickle.dump(dp, f)

    with open(args.mean_std, 'wb') as f:
        pickle.dump({
            'mean': dp.mean,
            'std': dp.std}, f)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
