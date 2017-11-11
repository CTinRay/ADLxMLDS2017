import argparse
import pdb
import pickle
import sys
import traceback
from pytorch_s2vt import TorchS2VT


def main():
    parser = argparse.ArgumentParser(description='ADL HW2')
    parser.add_argument('pickle', type=str,
                        help='Pickle made by make_pickle.py')
    parser.add_argument('model_path', type=str,
                        help='Path for model')
    parser.add_argument('data_path', type=str,
                        help='Path for raw data directory')
    parser.add_argument('predict', type=str,
                        help='Predict file')
    parser.add_argument('--arch', type=str,
                        help='model type', default='s2vt')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=16)
    args = parser.parse_args()

    with open(args.pickle, 'rb') as f:
        data_processor = pickle.load(f)

    # get dataset
    test = data_processor.get_test_dataset(args.data_path)
                                           # vids=['klteYv1Uv9A_27_33.avi',
                                           #       '5YJaS2Eswg0_22_26.avi',
                                           #       'UbmZAe5u5FI_132_141.avi',
                                           #       'JntMAcTlOF0_50_70.avi',
                                           #       'tJHUH9tpqPg_113_118.avi'])

    frame_dim = data_processor.get_frame_dim()
    word_dim = data_processor.get_word_dim()

    # select model arch
    archs = {
        's2vt': TorchS2VT}

    # init classifier
    clf = archs[args.arch](frame_dim=frame_dim,
                           word_dim=word_dim,
                           batch_size=args.batch_size)

    clf.load(args.model_path)
    test_y_ = clf.predict_dataset(test,
                                  predict_fn=clf._beam_search_batch)

    vids = [data['id'] for data in test]
    data_processor.write_predict(vids, test_y_, args.predict)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
