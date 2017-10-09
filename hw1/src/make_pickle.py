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
    parser.add_argument('--test_only', type=bool, default=False,
                        help='Whether or not only read test data.')
    args = parser.parse_args()

    print('Processing data.', file=sys.stderr)
    dp = DataProcessor(args.path)

    print('Start dumping pickle.', file=sys.stderr)
    with open(args.out, 'wb') as f:
        pickle.dump(dp, f)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
