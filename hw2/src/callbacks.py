import math
import numpy as np


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
            print('id = {}, predict = {}'.format(
                data[i]['id'],
                self._data_processor.indices_to_sentence(predicts[i])))
