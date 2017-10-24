import argparse
import pdb
import pickle
import sys
import traceback
import editdistance
import numpy as np
from rnn import RNNClassifier
from rnn_cnn import RNNCNNClassifier


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
    parser.add_argument('raw_data_path', type=str,
                        help='Path for raw data directory')
    parser.add_argument('predict', type=str, help='Predict file')
    parser.add_argument('--model', type=str,
                        help='model type', default='rnn')
    parser.add_argument('--gpu_mem', type=float,
                        help='GPU memory fraction', default=0.1)
    args = parser.parse_args()

    with open(args.data, 'rb') as f:
        data_processor = pickle.load(f)

    train, valid = data_processor.get_train_valid()
    classifiers = {
        'rnn': RNNClassifier,
        'rnncnn': RNNCNNClassifier}

    n_classes = np.max(train['y']) + 1
    clf = classifiers[args.model](train['x'].shape, n_classes,
                                  gpu_memory_fraction=args.gpu_mem)
    clf.load(args.path)

    valid['y_'] = clf.predict(valid['x'])
    valid['seqs'] = data_processor.int_to_char(valid['y'], valid['x'])
    valid['seqs_'] = data_processor.int_to_char(valid['y_'], valid['x'])

    print('mean distance = %f', mean_distance(valid['seqs'], valid['seqs_']))

    test = data_processor.get_test()
    test['y_'] = clf.predict(test['x'])
    test['seqs_'] = data_processor.int_to_char(test['y_'], test['x'])
    data_processor.write_predict(test['seqs_'],
                                 args.raw_data_path,
                                 args.predict)
    pdb.set_trace()


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
