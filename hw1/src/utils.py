import os
import pickle
import re
import pandas as pd
import numpy as np
import pdb


class DataProcessor:
    def _get_data(self, filename,
                  preserve_order=False):
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

        if preserve_order:
            sentence_ids_unique = [sentence_ids[0]]
            for sid in sentence_ids[1:]:
                if sid != sentence_ids_unique[-1]:
                    sentence_ids_unique.append(sid)
            df = df.reindex(sentence_ids_unique)

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

    def __init__(self, path, test_only=False, mean_var_file=None):

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

        self.phone_char_map = [''] * 39
        phone_char_file = os.path.join(path, '48phone_char.map')
        with open(phone_char_file) as f:
            for row in f:
                phone = row.split()[0]
                char = row.split()[2]
                if phone in self.phone_map39:
                    self.phone_char_map[self.phone_map48[phone]] = char

        if not test_only:
            self.train = {}
            self.train['x'] = \
                self._get_data(os.path.join(path, 'fbank', 'train.ark'))
            self.train['y'] = \
                self._get_label(os.path.join(path, 'label', 'train.lab'))

            # self.train['x'] = np.concatenate(
            #     (self.train['x'],
            #      np.diff(self.train['x'], 1, -1),
            #      np.diff(self.train['x'], 2, -1)),
            #     axis=-1)

            phones = self.train['x'].reshape((-1, self.train['x'].shape[-1]))
            phones = phones[~np.isnan(phones[:, 1])]
            self.mean = np.mean(phones, axis=0)
            self.std = np.std(phones, axis=0)

            self.train['x'] = (self.train['x'] - self.mean) / self.std

            indices = np.arange(self.train['x'].shape[0])
            np.random.shuffle(indices)
            self.train['x'] = self.train['x'][indices]
            self.train['y'] = self.train['y'][indices]
        else:
            with open(mean_var_file, 'rb') as f:
                mean_var = pickle.load(f)
                self.mean = mean_var['mean']
                self.std = mean_var['std']

        self.test = {}
        self.test['x'] = \
            self._get_data(os.path.join(path, 'fbank', 'test.ark'),
                           preserve_order=True)
        # self.test['x'] = np.concatenate(
        #         (self.test['x'],
        #          np.diff(self.test['x'], 1, -1),
        #          np.diff(self.test['x'], 2, -1)),
        #         axis=-1)
        self.test['x'] = (self.test['x'] - self.mean) / self.std

    def get_train_valid(self, valid_ratio=0.2):
        n_valid = int(self.train['x'].shape[0] * valid_ratio)
        train = {'x': self.train['x'][n_valid:],
                 'y': self.train['y'][n_valid:]}
        valid = {'x': self.train['x'][:n_valid],
                 'y': self.train['y'][:n_valid]}
        return train, valid

    def get_test(self):
        return self.test

    def int_to_char(self, pred, x):

        # convert ints pred to chars
        chars = np.array(self.phone_char_map)[pred]

        # convert to char string
        lengths = np.sum(1 - np.isnan(x[:, :, 0]).astype(int), axis=-1)
        seqs = list(map(lambda x, l: ''.join(x[:l]), chars, lengths))

        # remove consecutive
        pattern = re.compile(r'([a-zA-Z])\1+')
        seqs = list(map(lambda x: pattern.sub(r'\1', x), seqs))

        # trim silence
        silence = self.phone_map48['sil']
        silence_char = self.phone_char_map[silence]
        pattern = re.compile('(^%s*|%s*$)' % (silence_char, silence_char))
        seqs = list(map(lambda x: pattern.sub('', x), seqs))

        return seqs

    def write_predict(self, seqs, path, out):
        df = pd.read_csv(os.path.join(path, 'mfcc', 'test.ark'),
                         delim_whitespace=True,
                         header=None)
        ids = df[0]
        sentence_ids = \
            list(map(lambda x: '_'.join(x.split('_')[:-1]), ids))
        sentence_ids_unique = [sentence_ids[0]]

        for sid in sentence_ids[1:]:
            if sid != sentence_ids_unique[-1]:
                sentence_ids_unique.append(sid)

        df = pd.DataFrame(seqs, index=sentence_ids_unique)
        df.index.name = 'id'
        df.to_csv(out, header=['phone_sequence'])
