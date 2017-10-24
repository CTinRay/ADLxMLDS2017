import os
import sys


data_path = sys.argv[1]
os.system('python3 src/make_pickle.py '
          '{data_path} data-processor.pickle mean_std.pickle'
          .format(data_path=data_path))
os.system('mkdir -p ../model/rnn')
os.system('python3 src/train.py '
          'data-processor.pickle ../model/rnn '
          '--batch_size 32 --gpu_mem 1 --model rnn'
          .format(data_path=data_path))
