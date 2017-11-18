import re
import json
import os
import torch
import numpy as np
from bleu_eval import BLEU
from torch.utils.data import Dataset
import pdb
import sys


class MSVDDataset(Dataset):
    def __init__(self, data_dir,
                 mean, std, labels=None, vids=None):
        self._base = os.path.join(data_dir, 'feat')
        self._files = os.listdir(self._base)
        self._labels = labels
        self._mean = mean
        self._std = std

        if vids is not None:
            self._files = [vid + '.npy' for vid in vids]

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        item = {}
        item['x'] = np.load(os.path.join(self._base, self._files[idx]))
        item['x'] = item['x'].astype(np.float32)
        # item['x'] = item['x'][::-1]
        item['video_len'] = np.sum(np.sum(item['x'], axis=-1) > 0)
        item['video_mask'] = torch.arange(0, item['x'].shape[0]) < float(item['video_len'])
        item['video_mask'] = item['video_mask'].float()
        item['x'] = (item['x'] - self._mean) \
            / (self._std + sys.float_info.epsilon)

        vid = self._files[idx].replace('.npy', '')
        item['id'] = vid

        if self._labels is not None:
            n_captions = len(self._labels[vid])
            cap_index = np.random.randint(0, n_captions)
            item['y'] = self._labels[vid][cap_index]
            item['caption_len'] = len(item['y'])
            # item['lengths'] = list(map(len, self._labels[idx][cap_index]))

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
        self._dict = {'<pad>': 0, '<eos>': 1, '<sos>': 2, '<unk>': 3}
        self._word_list = ['<pad>', '<eos>', '<sos>', '<unk>']
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
        sentence = '<sos> ' + sentence + ' <eos>'
        tokens = self._tokenize(sentence)
        indices = list(map(lambda token: self._dict[token]
                           if token in self._dict else 3, tokens))
        return indices

    def _json_obj_to_dict(self, jobj):
        dictionary = {}
        for label in jobj:
            vid = label['id']
            dictionary[vid] = [
                self._sentence_to_indices(caption)
                for caption in label['caption']]

        return dictionary

    def _calculate_mean_std(self, dataset):
        xs = [data['x'] for data in dataset]
        xs = np.concatenate(xs, axis=0)
        self._mean = np.mean(xs, 0)
        self._std = np.std(xs, 0)

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

        self._calculate_mean_std(
            MSVDDataset(os.path.join(path, 'training_data'), 0, 1,
                        self.train_labels))

    def get_train_dataset(self, path):
        return MSVDDataset(os.path.join(path, 'training_data'),
                           self._mean, self._std,
                           self.train_labels)

    def get_test_dataset(self, path, vids=None):
        return MSVDDataset(os.path.join(path, 'testing_data'),
                           self._mean, self._std,
                           self.test_labels, vids)

    def get_word_dim(self):
        return len(self._word_list)

    def get_frame_dim(self):
        return 4096

    def indices_to_sentence(self, indices):
        return ' '.join([self._word_list[index] for index in indices])

    def postprocess_sentence(self, sentence):
        sentence = sentence.replace('<sos> ', '') \
          .replace(' <pad>', '') \
          .replace('.', '')
        sentence = sentence[:sentence.find('<eos>')]
        return sentence

    def write_predict(self, vids, predicts, filename):
        sentences = \
            [self.indices_to_sentence(predict) for predict in predicts]
        sentences = \
            [self.postprocess_sentence(sentence) for sentence in sentences]
        with open(filename, 'w') as f:
            for vid, sentence in zip(vids, sentences):
                f.write("%s,%s\n" % (vid, sentence))


def calc_bleu(predict, labels):
    predict = ' '.join(map(str, predict))
    labels = [' '.join(map(str, label)) for label in labels]
    return BLEU(predict, labels)
