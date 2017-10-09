import argparse
import pdb
import pickle
import sys
import traceback
import numpy as np
from rnn import RNNClassifier


def main():
    parser = argparse.ArgumentParser(description='ADL HW1')
    parser.add_argument('data', type=str, help='Pickle made by make_pickle.py')
    # parser.add_argument('out', type=str, help='Filename of output pickle')
    parser.add_argument('--valid_ratio', type=float,
                        help='Ratio of validation data', default=0.2)
    parser.add_argument('--n_epochs', type=int,
                        help='Number of epochs', default=100)
    parser.add_argument('--lr', type=float,
                        help='Learning rate', default=1e-5)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=128)
    parser.add_argument('--model', type=str,
                        help='model type', default='rnn')
    args = parser.parse_args()

    with open(args.data, 'rb') as f:
        data_processor = pickle.load(f)

    train, valid = data_processor.get_train_valid()
    # test = data_processor.get_test()

    n_classes = np.max(train['y']) + 1
    classifiers = {'rnn': RNNClassifier(train['x'].shape,
                                        n_classes,
                                        valid=valid)}
    clf = classifiers[args.model]

    clf.fit(train['x'], train['y'])


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
