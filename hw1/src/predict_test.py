import argparse
import pdb
import pickle
import sys
import traceback
import editdistance
import numpy as np
from rnn import RNNClassifier
from rnn_cnn import RNNCNNClassifier
from utils import DataProcessor


def mean_distance(truth, pred):
    return np.mean(
        list(
            map(lambda t, p: editdistance.eval(t, p),
                truth, pred)
        )
    )


def main():
    parser = argparse.ArgumentParser(description='ADL HW1')
    parser.add_argument('data_path', type=str,
                        help='Path for raw data directory')
    parser.add_argument('mean_std_pickle', type=str,
                        help='Path for mean-std.picle')
    parser.add_argument('model_path', type=str, help='Path for model')
    parser.add_argument('predict', type=str, help='Predict file')
    parser.add_argument('--model', type=str,
                        help='model type', default='rnn')
    parser.add_argument('--gpu_mem', type=float,
                        help='GPU memory fraction', default=0.2)
    parser.add_argument('--feature', type=str,
                        help='mfcc or fbank', default='mfcc')
    args = parser.parse_args()

    data_processor = DataProcessor(args.data_path,
                                   test_only=True,
                                   mean_var_file=args.mean_std_pickle,
                                   feature=args.feature)

    test = data_processor.get_test()
    classifiers = {
        'rnn': RNNClassifier,
        'rnncnn': RNNCNNClassifier}

    n_classes = 39
    clf = classifiers[args.model](test['x'].shape, n_classes,
                                  gpu_memory_fraction=args.gpu_mem)
    clf.load(args.model_path)

    test['y_'] = clf.predict(test['x'])
    test['seqs_'] = data_processor.int_to_char(test['y_'], test['x'])
    data_processor.write_predict(test['seqs_'],
                                 args.data_path,
                                 args.predict)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
