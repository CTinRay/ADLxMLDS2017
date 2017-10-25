import math
import editdistance
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


class EditDistance(Callback):
    def _mean_distance(self, truth, pred):
        return np.mean(
            list(
                map(lambda t, p: editdistance.eval(t, p),
                    truth, pred)
            )
        )

    def __init__(self, valid, data_processor):
        self._valid = valid
        self._dp = data_processor
        self._seqs = data_processor.int_to_char(valid['y'], valid['x'])

    def on_epoch_end(self, log_train, log_valid, model):
        y_ = model.predict(self._valid['x'])
        seqs_ = self._dp.int_to_char(y_, self._valid['x'])
        print('editdistance %f' % self._mean_distance(self._seqs, seqs_))
