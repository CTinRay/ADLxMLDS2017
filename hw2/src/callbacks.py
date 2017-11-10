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

    def on_epoch_end(self, log_train, log_valid, model):
        ys_ = model.predict_dataset(self._dataset)
        sentences = [' '.join(map(lambda n: str(n) if n > 3 else '',
                                  y.tolist()))
                     for y in ys_]

        bleu = 0
        for sentence, data in zip(sentences, self._dataset):
            sentence_bleu = 0
            for y in self._data_processor.test_labels[data['id']]:
                ans = ' '.join(map(str, y[1:-1]))
                sentence_bleu += BLEU(sentence, ans)

            sentence_bleu /= len(self._data_processor.test_labels[data['id']])
            bleu += sentence_bleu

        bleu /= len(self._dataset)
        log_valid['bleu'] = bleu

        print('BLEU score = %f' % bleu)
