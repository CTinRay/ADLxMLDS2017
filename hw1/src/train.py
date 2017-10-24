import argparse
import pdb
import pickle
import sys
import traceback
import numpy as np
from callbacks import ModelCheckpoint
from rnn import RNNClassifier
from rnn_cnn import RNNCNNClassifier
from utils import DataProcessor


def main():
    parser = argparse.ArgumentParser(description='ADL HW1')
    parser.add_argument('data', type=str, help='Pickle made by make_pickle.py')
    parser.add_argument('path', type=str, help='Path for model')
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
    parser.add_argument('--gpu_mem', type=float,
                        help='GPU memory fraction', default=0.1)
    args = parser.parse_args()

    with open(args.data, 'rb') as f:
        data_processor = pickle.load(f)
    # data_processor = DataProcessor(args.data)

    train, valid = data_processor.get_train_valid()
    # test = data_processor.get_test()

    classifiers = {
        'rnn': RNNClassifier,
        'rnncnn': RNNCNNClassifier}

    n_classes = np.max(train['y']) + 1
    clf = classifiers[args.model](train['x'].shape, n_classes,
                                  batch_size=args.batch_size,
                                  valid=valid,
                                  n_epochs=args.n_epochs,
                                  gpu_memory_fraction=args.gpu_mem)

    model_checkpoint = ModelCheckpoint(args.path,
                                       'accuracy', 1, 'max')
    clf.fit(train['x'], train['y'],
            [model_checkpoint])


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
