import os
import random
import numpy as np
import skimage.io
import skimage.transform
import torch
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    def __init__(self, img_dir, labels):
        super(ImgDataset, self).__init__()
        self._img_dir = img_dir
        self._labels = labels

    def __len__(self):
        return len(self._labels)

    def _get_img(self, index):
        filename = os.path.join(self._img_dir, '%d.jpg' % index)
        img = skimage.io.imread(filename)
        img = img / 128 - 1.0
        img = np.transpose(img, [2, 0, 1]).astype(np.float32)
        return img


class FakeDataset(ImgDataset):
    def __init__(self, img_dir, labels, check_fake_fn):
        super(FakeDataset, self).__init__(img_dir, labels)
        self._check_fake_fn = check_fake_fn

    def __getitem__(self, index):
        truth_label = self._labels[index]
        rand_index = random.randrange(len(self._labels))
        fake_label = self._labels[rand_index]
        while not self._check_fake_fn(truth_label, fake_label):
            rand_index = random.randrange(len(self._labels))
            fake_label = self._labels[rand_index]

        img = self._get_img(index)
        item = {'img': img,
                'label': fake_label}
        return item


class RealDataset(ImgDataset):
    def __init__(self, img_dir, labels):
        super(RealDataset, self).__init__(img_dir, labels)

    def __getitem__(self, index):
        item = {
            'img': self._get_img(index),
            'label': self._labels[index]}
        return item


class DataProcessor:
    def __init__(self, img_dir, tag_file):
        self._img_dir = img_dir
        self._hair_tag_dict = {
            'orange hair': 0, 'white hair': 1, 'aqua hair': 2,
            'gray hair': 3, 'green hair': 4, 'red hair': 5,
            'purple hair': 6, 'pink hair': 7, 'blue hair': 8,
            'black hair': 9, 'brown hair': 10, 'blonde hair': 11}
        self._eye_tag_dict = {
             'gray eyes': 0, 'black eyes': 1, 'orange eyes': 2,
             'pink eyes': 3, 'yellow eyes': 4, 'aqua eyes': 5,
             'purple eyes': 6, 'green eyes': 7, 'brown eyes': 8,
             'red eyes': 9, 'blue eyes': 10}
        img_tags = self._read_tags(tag_file)
        self._img_labels = self._encode_tags(img_tags)

    def _read_tags(self, tag_file):
        img_tags = []
        with open(tag_file) as f:
            for l in f:
                tags = l.strip().split(',')[1].split('\t')
                tags = [tag.split(':')[0] for tag in tags]
                img_tags.append(tags)
        return img_tags

    def _encode_tags(self, img_tags):
        labels = torch.zeros(
            len(img_tags),
            len(self._eye_tag_dict) + len(self._hair_tag_dict))
        for i, img_tag in enumerate(img_tags):
            for tag in img_tag:
                if tag in self._hair_tag_dict:
                    labels[i, self._hair_tag_dict[tag]] = 1
                elif tag in self._eye_tag_dict:
                    labels[i,
                           len(self._hair_tag_dict)
                           + self._eye_tag_dict[tag]] = 1

        return labels

    def _make_tag_dict(self, img_tags):
        self.hair_tag_dict = {}
        self.eye_tag_dict = {}
        for tags in img_tags:
            for tag in tags:
                if 'hair' in tag and \
                   tag not in self.hair_tag_dict:
                    self.hair_tag_dict[tag] = len(self.hair_tag_dict)
                elif 'eye' in tag and \
                     tag not in self.eye_tag_dict:
                    self.eye_tag_dict[tag] = len(self.eye_tag_dict)

    def check_fake_label(self, truth_label, label):
        return label.sum() == 2 and (truth_label != label).any()
        # # check if hair tag is different
        # truth_hair_label = truth_label[:len(self._hair_tag_dict)]
        # hair_label = label[:len(self._hair_tag_dict)]
        # if truth_hair_label.sum() > 0 and \
        #    hair_label.sum() > 0 and \
        #    truth_hair_label @ hair_label == 0:
        #     return True

        # # check if eye tag is different
        # truth_eye_label = truth_label[len(self._hair_tag_dict):]
        # eye_label = label[len(self._hair_tag_dict):]
        # if truth_eye_label.sum() > 0 and \
        #    eye_label.sum() > 0 and \
        #    truth_eye_label @ hair_label == 0:
        #     return True

        # return False

    def get_fake_dataset(self):
        return FakeDataset(self._img_dir, self._img_labels,
                           self.check_fake_label)

    def get_real_dataset(self):
        return RealDataset(self._img_dir, self._img_labels)

    def get_dim_condition(self):
        return len(self._eye_tag_dict) + len(self._hair_tag_dict)
