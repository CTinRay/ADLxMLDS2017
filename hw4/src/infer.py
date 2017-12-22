import argparse
import os
import numpy as np
import skimage.io
from utils import DataProcessor
from gan import GAN
import pdb
import sys
import traceback


def main(args):
    dim_condition = 23
    gan = GAN(dim_condition)
    gan.load(args.model)

    conditions = np.zeros((12 * 11 * 5, dim_condition))
    for i in range(12 * 11 * 5):
        h = (i // 55) % 12
        e = (i // 5) % 11
        n = (i // 1) % 5
        conditions[i, h] = 1
        conditions[i, 12 + e] = 1
    imgs = gan.inference(conditions)
    for i in range(12 * 11 * 5):
        h = (i // 55) % 12
        e = (i // 5) % 11
        n = (i // 1) % 5
        filename = os.path.join(args.out_dir,
                                '{}-{}-{}.jpg'.format(h, e, n))
        skimage.io.imsave(filename, imgs[i])


def _parse_args():
    parser = argparse.ArgumentParser(description="ADL HW4")
    parser.add_argument('model', type=str)
    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    try:
        main(args)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
