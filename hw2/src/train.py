import argparse
import pdb
import pickle
import sys
import traceback
from callbacks import ModelCheckpoint, PrintPredict, CalcBleu
from pytorch_s2vt import TorchS2VT


def parse_args():
    parser = argparse.ArgumentParser(description='ADL HW2')
    parser.add_argument('pickle', type=str,
                        help='Pickle made by make_pickle.py')
    parser.add_argument('data_path', type=str, help='Directory of the data.')
    parser.add_argument('model_path', type=str, help='Path of the model')
    parser.add_argument('--n_epochs', type=int,
                        help='Number of epochs', default=100)
    parser.add_argument('--lr', type=float,
                        help='Learning rate', default=1e-5)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=128)
    parser.add_argument('--arch', type=str,
                        help='Network architecture', default='s2vt')
    parser.add_argument('--gpu_mem', type=float,
                        help='GPU memory fraction', default=0.1)
    args = parser.parse_args()
    return args


def main(args):
    # load data_processor
    with open(args.pickle, 'rb') as f:
        data_processor = pickle.load(f)

    # get dataset
    train = data_processor.get_train_dataset(args.data_path)
    test = data_processor.get_test_dataset(args.data_path)

    frame_dim = data_processor.get_frame_dim()
    word_dim = data_processor.get_word_dim()

    # select model arch
    archs = {
        's2vt': TorchS2VT}

    # init classifier
    clf = archs[args.arch](frame_dim=frame_dim,
                           word_dim=word_dim,
                           batch_size=args.batch_size,
                           valid=test,
                           n_epochs=args.n_epochs)

    # make callbacks
    model_checkpoint = ModelCheckpoint(args.model_path,
                                       'bleu', 1, 'max')
    print_predict_train = PrintPredict(train, data_processor)
    print_predict_test = PrintPredict(test, data_processor)
    calc_bleu = CalcBleu(test, data_processor)

    # fit
    clf.fit_dataset(train, [calc_bleu,
                            print_predict_train,
                            print_predict_test,
                            model_checkpoint])


if __name__ == '__main__':
    try:
        args = parse_args()
        main(args)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
