import numpy as np
import torch
from torch.autograd import Variable
from pytorch_base import TorchBase
from s2vt import S2VT


class TorchS2VT(TorchBase):
    def __init__(self, frame_dim, word_dim, **kwargs):
        super(TorchS2VT, self).__init__(**kwargs)
        self._word_dim = word_dim
        self._model = S2VT(frame_dim, word_dim)

        # make class weights to ignore loss for padding
        class_weights = torch.ones(word_dim)
        class_weights[0] = 0

        # use cuda
        if self._use_cuda:
            class_weights = class_weights.cuda()
            self._model = self._model.cuda()

        # make loss
        self._loss = torch.nn.CrossEntropyLoss(class_weights,
                                               size_average=False)

        # make optimizer
        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=self._learning_rate)

    def _run_iter(self, batch, training):
        var_x = Variable(batch['x'].transpose(0, 1))
        var_y = [Variable(y) for y in batch['y']]
        batch['video_len'] = batch['video_len'].tolist()
        batch['caption_len'] = batch['caption_len'].tolist()

        if self._use_cuda:
            var_x = var_x.cuda()
            var_y = [vy.cuda() for vy in var_y]

        # encode
        hidden_video, hidden_caption = \
            self._model.encoder.forward(var_x,
                                        batch['video_len'],
                                        training)

        var_loss = 0
        batch_size = batch['x'].shape[0]

        var_predicts = Variable(torch.ones(1, batch_size).long())
        var_ones = Variable(torch.ones(batch_size))
        if self._use_cuda:
            var_predicts = var_predicts.cuda()
            var_ones = var_ones.cuda()

        for i in range(1, max(batch['caption_len'])):
            # flip coins with teach_prob
            teach_prob = 1 - (self._epoch / (150 + self._epoch)) \
                         if training else 0
            if_teach = torch.bernoulli(teach_prob * var_ones)
            if_teach = if_teach.long()

            # take previous label according to if_teach
            prev_word = var_y[i - 1] * if_teach \
                + var_predicts[-1, :] * (1 + if_teach * -1)

            prev_word = prev_word.detach()
            # decode
            probs, hidden_video, hidden_caption = \
                self._model.decoder.forward(prev_word,
                                            hidden_video, hidden_caption,
                                            training)

            # store prediction
            var_predicts = \
                torch.cat([var_predicts,
                           torch.max(probs, -1)[1].unsqueeze(0)],
                          0)

            # accumulate loss
            var_loss += self._loss(probs, var_y[i])

        var_loss = var_loss / sum(batch['caption_len'])
        return var_predicts, var_loss

    def _predict_batch(self, batch):
        var_x = Variable(batch['x'].transpose(0, 1))
        batch['video_len'] = batch['video_len'].tolist()

        if self._use_cuda:
            var_x = var_x.cuda()

        # encode
        hidden_video, hidden_caption = \
            self._model.encoder.forward(var_x,
                                        batch['video_len'],
                                        False)

        batch_size = batch['x'].shape[0]

        var_predicts = Variable(torch.ones(1, batch_size).long())
        if self._use_cuda:
            var_predicts = var_predicts.cuda()

        for i in range(1, max(batch['caption_len'])):
            # take previous label according to if_teach
            prev_word = var_predicts[-1, :]

            prev_word = prev_word.detach()
            # decode
            probs, hidden_video, hidden_caption = \
                self._model.decoder.forward(prev_word,
                                            hidden_video, hidden_caption,
                                            False)

            # store prediction
            var_predicts = \
                torch.cat([var_predicts,
                           torch.max(probs, -1)[1].unsqueeze(0)],
                          0)

        return var_predicts.data.cpu().numpy().T
