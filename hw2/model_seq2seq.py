import os
import sys


data_path = sys.argv[1]
os.system('mkdir -p models')
os.system('python3 src/make_pickle.py '
          '{data_path} models/data-processor.pickle'
          .format(data_path=data_path))
os.system('python3 src/train.py '
          'models/data-processor.pickle {data_path} models/hLSTMat '
          '--batch_size 64 --arch hLSTMat --n_epochs 200'
          .format(data_path=data_path))
