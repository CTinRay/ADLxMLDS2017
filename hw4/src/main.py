import argparse
from utils import DataProcessor
from gan import GAN
import pdb
import sys
import traceback


def main(args):
    data_processor = DataProcessor(args.img_dir, args.tag_file)
    dim_condition = data_processor.get_dim_condition()
    gan = GAN(dim_condition, save_dir=args.save_dir, max_epochs=500)
    gan.train(data_processor.get_real_dataset(),
              data_processor.get_fake_dataset())


def _parse_args():
    parser = argparse.ArgumentParser(description="ADL HW4")
    parser.add_argument('img_dir', type=str,
                        help='Directory that contains face images.')
    parser.add_argument('tag_file', type=str,
                        help='File that contains tags of the images')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='Place where checkpoint will be saved.')
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
