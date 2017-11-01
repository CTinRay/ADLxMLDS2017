import argparse
import pdb
import pickle
import sys
import traceback
import numpy as np
from callbacks import ModelCheckpoint
from pytorch_wrapper import TorchWrapper
from s2vd import S2VD
import torch.nn


def main():
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
                        help='Network architecture', default='s2vd')
    parser.add_argument('--gpu_mem', type=float,
                        help='GPU memory fraction', default=0.1)
    args = parser.parse_args()

    with open(args.pickle, 'rb') as f:
        data_processor = pickle.load(f)

    train = data_processor.get_train_dataset(args.data_path)
    test = data_processor.get_test_dataset(args.data_path)

    archs = {
        's2vd': S2VD}

    frame_dim = data_processor.get_frame_dim()
    word_dim = data_processor.get_word_dim()
    class_weights = torch.ones(word_dim)
    class_weights[0] = 0
    clf = TorchWrapper(archs[args.arch](frame_dim, word_dim),
                       torch.nn.CrossEntropyLoss(class_weights),
                       batch_size=args.batch_size,
                       valid=test,
                       n_epochs=args.n_epochs,
                       gpu_memory_fraction=args.gpu_mem)

    model_checkpoint = ModelCheckpoint(args.model_path,
                                       'loss', 1, 'min')
    clf.fit_dataset(train, [model_checkpoint])


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
