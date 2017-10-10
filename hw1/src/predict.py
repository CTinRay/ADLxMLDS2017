import argparse
import pdb
import pickle
import sys
import traceback
import editdistance
import numpy as np
from rnn import RNNClassifier


def mean_distance(truth, pred):
    return np.mean(
        list(
            map(lambda t, p: editdistance.eval(t, p),
                truth, pred)
        )
    )


def main():
    parser = argparse.ArgumentParser(description='ADL HW1')
    parser.add_argument('data', type=str, help='Pickle made by make_pickle.py')
    parser.add_argument('path', type=str, help='Path for model')
    parser.add_argument('--model', type=str,
                        help='model type', default='rnn')
    args = parser.parse_args()

    with open(args.data, 'rb') as f:
        data_processor = pickle.load(f)

    train, valid = data_processor.get_train_valid()
    classifiers = {'rnn': RNNClassifier(train['x'].shape,
                                        39,
                                        gpu_memory_fraction=1)}

    clf = classifiers[args.model]
    clf.load(args.path)

    valid['y_'] = clf.predict(valid['x'])
    valid['seqs'] = data_processor.int_to_char(valid['y'], valid['x'])
    valid['seqs_'] = data_processor.int_to_char(valid['y_'], valid['x'])

    print('mean distance = %f', mean_distance(valid['seqs'], valid['seqs_']))

    pdb.set_trace()


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
