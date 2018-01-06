import argparse
import os
import pdb
import sys
import traceback
import skimage.io
import numpy as np
from gan import GAN
from utils import DataProcessor


def main(args):
    dim_condition = 23
    gan = GAN(dim_condition,
              cuda_rand_seed=args.rand_seed)
    gan.load(args.model)

    data_processor = DataProcessor(None, None)
    tids, conds = data_processor.get_test_condition(args.test_file,
                                                    args.n_imgs_per_cond)
    imgs = gan.inference(conds)
    imgs = np.transpose(imgs, [0, 2, 3, 1])
    imgs = ((imgs + 1) * 128).astype('uint8')

    for t, tid in enumerate(tids):
        for i in range(args.n_imgs_per_cond):
            filename = os.path.join(args.out_dir,
                                    'sample-{}-{}.jpg'.format(tid, i + 1))
            skimage.io.imsave(filename, imgs[t * args.n_imgs_per_cond + i])


def _parse_args():
    parser = argparse.ArgumentParser(description="ADL HW4")
    parser.add_argument('model', type=str)
    parser.add_argument('test_file', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--n_imgs_per_cond', type=int, default=5)
    parser.add_argument('--rand_seed', type=int, default=0)
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
