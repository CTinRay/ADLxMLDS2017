import os
import pandas as pd
import numpy as np
import pdb


class DataProcessor:
    def _get_data(self, filename):
        df = pd.read_csv(filename,
                         delim_whitespace=True,
                         header=None)

        # extract sentence id and frame id from first column
        sentence_ids = \
            list(map(lambda x: '_'.join(x.split('_')[:-1]), df[0]))
        frame_ids = \
            list(map(lambda x: int(x.split('_')[-1]), df[0]))
        del df[0]

        # add sid, fid into df and pivot it accordingly
        df['sid'] = sentence_ids
        df['fid'] = frame_ids
        df = df.pivot('sid', 'fid')

        # swap and sort so the first level is fid and
        # the second is 69 features
        df.columns = df.columns.swaplevel(0, 1)
        df.sort_index(axis=1, level=0, inplace=True)

        # sort by sentence id
        df.sort_index(axis=0, inplace=True)

        # reuturn 3d matrix
        return df.as_matrix().reshape(df.shape[0],
                                      df.columns[-1][0],
                                      df.columns[-1][1])

    def _get_label(self, filename):
        df = pd.read_csv(filename,
                         header=None)

        # extract sentence id and frame id from first column
        sentence_ids = \
            list(map(lambda x: '_'.join(x.split('_')[:-1]), df[0]))
        frame_ids = \
            list(map(lambda x: int(x.split('_')[-1]), df[0]))
        del df[0]

        # add sid, fid into df and pivot it accordingly
        df['sid'] = sentence_ids
        df['fid'] = frame_ids
        df = df.pivot('sid', 'fid')

        # sort by sentence id
        df.sort_index(axis=0, inplace=True)

        # encode phones to integers
        for phone48 in self.phone_map48:
            df.replace(phone48, self.phone_map48[phone48], inplace=True)
        df.fillna(-1, inplace=True)

        return df.as_matrix().astype(int)

    def __init__(self, path, test_only=False):

        # make phone tables
        self.phone_map48 = {}
        self.phone_map39 = []
        phone_map_file = os.path.join(path, 'phones', '48_39.map')
        with open(phone_map_file) as f:
            for line in f:
                phone48 = line.strip().split()[0]
                phone39 = line.strip().split()[1]
                if phone39 not in self.phone_map39:
                    self.phone_map39.append(phone39)
                self.phone_map48[phone48] = self.phone_map39.index(phone39)

        if not test_only:
            self.train = {}
            self.train['x'] = \
                self._get_data(os.path.join(path, 'fbank', 'train.ark'))
            self.train['y'] = \
                self._get_label(os.path.join(path, 'label', 'train.lab'))

            indices = np.arange(self.train['x'].shape[0])
            np.random.shuffle(indices)
            self.train['x'] = self.train['x'][indices]
            self.train['y'] = self.train['y'][indices]

        self.test = {}
        self.test['x'] = \
            self._get_data(os.path.join(path, 'fbank', 'test.ark'))

    def get_train_valid(self, valid_ratio=0.2):
        n_valid = int(self.train['x'].shape[0] * valid_ratio)
        train = {'x': self.train['x'][n_valid:],
                 'y': self.train['y'][n_valid:]}
        valid = {'x': self.train['x'][:n_valid],
                 'y': self.train['y'][:n_valid]}
        return train, valid

    def get_test(self):
        return self.test
