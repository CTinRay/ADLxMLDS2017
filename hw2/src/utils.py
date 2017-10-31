import re
import json
import os
import torch
import numpy as np
from bleu_eval import BLEU
from torch.utils.data import Dataset


class MSVDDataset(Dataset):
    def __init__(self, data_dir, labels=None):
        self._ids = os.listdir(data_dir)
        self._labels = labels

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        item = {}
        item['x'] = np.load(self._ids[idx])

        if self._labels is not None:
            # n_captions = len(self._labels[idx])
            # cap_index = np.random.randint(0, n_captions)
            item['y'] = self._labels[idx]

        return item


class DataProcessor:
    def _filter(self, sentence):
        pattern = re.compile('[a-zA-Z,.;!\'" ]')
        return ''.join(pattern.findall(sentence))

    def _tokenize(self, sentence):
        punctuations = [',', '.', ';', '!', '\'', '"']

        # add space after punctuations
        for punc in punctuations:
            sentence = sentence.replace(punc, ' ' + punc)

        # Turn to lower case
        sentence = sentence.lower()

        # tokenize
        tokens = sentence.split()

        return tokens

    def _make_dict(self, labels):
        self._dict = {'': 0}
        self._word_list = ['']
        for label in labels:
            for caption in label['caption']:
                caption = self._filter(caption)
                tokens = self._tokenize(caption)

                # put into dictionary and word list
                for token in tokens:
                    if token not in self._dict:
                        self._dict[token] = len(self._dict)
                        self._word_list.append(token)

    def _sentence_to_indices(self, sentence):
        sentence = self._filter(sentence)
        tokens = self._tokenize(sentence)
        indices = list(map(lambda token: self._dict[token]
                           if token in self._dict else 0, tokens))
        return indices

    def _json_obj_to_dict(self, jobj):
        dictionary = {}
        for label in jobj:
            vid = label['id']
            dictionary[vid] = [
                self._sentence_to_indices(caption)
                for caption in label['caption']]

        return dictionary

    def __init__(self, path):
        # load training labels and make word dict/list
        train_label_filename = os.path.join(path, 'training_label.json')
        with open(train_label_filename) as f:
            train_labels = json.load(f)
        self._make_dict(train_labels)
        self.train_labels = self._json_obj_to_dict(train_labels)

        # loat test labels
        test_label_filename = os.path.join(path, 'testing_label.json')
        with open(test_label_filename) as f:
            test_labels = json.load(f)
        self.test_labels = self._json_obj_to_dict(test_labels)

    def get_train_dataset(self, path):
        return MSVDDataset(self,
                           os.path.join(path, 'training_data'),
                           self.train_labels)

    def get_test_dataset(self, path):
        return MSVDDataset(self,
                           os.path.join(path, 'testing_data'),
                           self.test_labels)


def calc_bleu(predict, labels):
    predict = ' '.join(map(str, predict))
    labels = [' '.join(map(str, label)) for label in labels]
    return BLEU(predict, labels)
