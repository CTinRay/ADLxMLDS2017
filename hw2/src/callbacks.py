import math
import numpy as np
from bleu_eval import BLEU
import pdb


class Callback:
    def __init__():
        pass

    def on_epoch_end(log_train, log_valid, model):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, filepath,
                 monitor='loss',
                 verbose=0,
                 mode='min'):
        self._filepath = filepath
        self._verbose = verbose
        self._monitor = monitor
        self._best = math.inf if mode == 'min' else - math.inf
        self._mode = mode

    def on_epoch_end(self, log_train, log_valid, model):
        score = log_valid[self._monitor]
        if self._mode == 'min':
            if score < self._best:
                self._best = score
                model.save(self._filepath)
                if self._verbose > 0:
                    print('Best model saved (%f)' % score)

        else:
            if score > self._best:
                self._best = score
                model.save(self._filepath)
                if self._verbose > 0:
                    print('Best model saved (%f)' % score)


class PrintPredict(Callback):
    def __init__(self, dataset, data_processor, n_samples=5):
        self._dataset = dataset
        self._data_processor = data_processor
        self._n_samples = n_samples

    def on_epoch_end(self, log_train, log_valid, model):
        data = [self._dataset[i] for i in range(self._n_samples)]
        predicts = model.predict_dataset(data)

        for i in range(predicts.shape[0]):
            print('id = {}, predict = {}, label = {}'.format(
                data[i]['id'],
                self._data_processor.indices_to_sentence(predicts[i]),
                self._data_processor.indices_to_sentence(data[i]['y'])))


class CalcBleu(Callback):
    def __init__(self, dataset, data_processor):
        self._dataset = dataset
        self._data_processor = data_processor
        self._period = data_processor._dict['.']
        self._captions = []
        for data in dataset:
            captions = self._data_processor.test_labels[data['id']]
            captions = [data_processor.indices_to_sentence(caption)
                        for caption in captions]
            captions = [data_processor.postprocess_sentence(caption)
                        for caption in captions]
            self._captions.append(captions)

    def on_epoch_end(self, log_train, log_valid, model):
        ys_ = model.predict_dataset(self._dataset)
        sentences = [self._data_processor.indices_to_sentence(y)
                     for y in ys_]
        sentences = [self._data_processor.postprocess_sentence(sentence)
                     for sentence in sentences]

        bleu = 0
        for sentence, captions in zip(sentences, self._captions):
            sentence_bleu = 0
            for y in captions:
                sentence_bleu += BLEU(sentence, y)

            sentence_bleu /= len(captions)
            bleu += sentence_bleu

        bleu /= len(self._dataset)
        log_valid['bleu'] = bleu

        print('BLEU score = %f' % bleu)
