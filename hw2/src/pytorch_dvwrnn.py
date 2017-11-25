import pdb
import torch
from torch.autograd import Variable
from pytorch_base import TorchBase
from dvwrnn import DVWRNN


class TorchDVWRNN(TorchBase):
    def __init__(self, frame_dim, word_dim, **kwargs):
        super(TorchDVWRNN, self).__init__(**kwargs)
        self._word_dim = word_dim
        self._model = DVWRNN(frame_dim, word_dim)

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
                                           lr=0.001)

    def _run_iter(self, batch, training):
        var_x = Variable(batch['x'].transpose(0, 1))
        var_y = [Variable(y) for y in batch['y']]
        batch['video_mask'] = Variable(batch['video_mask']
                                       .transpose(0, 1)
                                       .unsqueeze(-1))
        batch['video_len'] = batch['video_len'].tolist()
        batch['caption_len'] = batch['caption_len'].tolist()

        if self._use_cuda:
            var_x = var_x.cuda()
            var_y = [vy.cuda() for vy in var_y]
            batch['video_mask'] = batch['video_mask'].cuda()

        # encode
        hidden_v, video_outputs = \
            self._model.encoder.forward(var_x,
                                        batch['video_len'],
                                        batch['video_mask'])
        hidden_w = None
        hidden_gate = None

        var_loss = 0
        batch_size = batch['x'].shape[0]

        var_predicts = Variable(torch.ones(1, batch_size).long() * 2)
        var_ones = Variable(torch.ones(batch_size))
        if self._use_cuda:
            var_predicts = var_predicts.cuda()
            var_ones = var_ones.cuda()

        for i in range(1, max(batch['caption_len'])):
            # flip coins with teach_prob
            # teach_prob = 1 - (1 * self._epoch / (100 + 3 * self._epoch)) \
            #     if training else 1
            teach_prob = 1
            if_teach = torch.bernoulli(teach_prob * var_ones)
            if_teach = if_teach.long()

            # take previous label according to if_teach
            prev_word = var_y[i - 1] * if_teach \
                + var_predicts[-1, :] * (1 + if_teach * -1)

            prev_word = prev_word.detach()
            # decode
            logits, hidden_v, hidden_w, hidden_gate = \
                self._model.decoder.forward(video_outputs, batch['video_mask'],
                                            prev_word,
                                            hidden_v, hidden_w, hidden_gate)

            probs = torch.nn.functional.softmax(logits)
            # store prediction
            var_predicts = \
                torch.cat([var_predicts,
                           torch.max(probs, 1)[1].view(1, -1)],
                          0)

            # accumulate loss
            var_loss += self._loss(logits, var_y[i])

        var_loss = var_loss / sum(batch['caption_len'])
        return var_predicts, var_loss

    def _predict_batch(self, batch, max_len=30):
        var_x = Variable(batch['x'].transpose(0, 1), volatile=True)
        batch['video_len'] = batch['video_len'].tolist()
        batch['video_mask'] = Variable(batch['video_mask']
                                       .transpose(0, 1)
                                       .unsqueeze(-1))

        if self._use_cuda:
            var_x = var_x.cuda()
            batch['video_mask'] = batch['video_mask'].cuda()

        # encode
        hidden1, hidden2 = \
            self._model.encoder.forward(var_x,
                                        batch['video_mask'])

        batch_size = batch['x'].shape[0]

        var_predicts = Variable(torch.ones(1, batch_size).long() * 2)
        if self._use_cuda:
            var_predicts = var_predicts.cuda()

        if_end = (var_predicts[-1] == 1).data

        predict_len = 0
        while not if_end.all() and predict_len < max_len:
            # take previous label according to if_teach
            prev_word = var_predicts[-1, :]

            prev_word = prev_word.detach()
            # decode
            probs, hidden1, hidden2 = \
                self._model.decoder.forward(var_x, batch['video_mask'],
                                            prev_word,
                                            hidden1, hidden2)

            # store prediction
            var_predicts = \
                torch.cat([var_predicts,
                           torch.max(probs, -1)[1].unsqueeze(0)],
                          0)

            if_end = if_end | (var_predicts[-1] == 1).data
            predict_len += 1

        return var_predicts.data.cpu().numpy().T

    def _beam_search_batch(self, batch, beam_size=5, max_len=30):
        var_x = Variable(batch['x'].transpose(0, 1), volatile=True)
        batch['video_len'] = batch['video_len'].tolist()
        batch['video_mask'] = Variable(batch['video_mask']
                                       .transpose(0, 1)
                                       .unsqueeze(-1))

        if self._use_cuda:
            var_x = var_x.cuda()
            batch['video_mask'] = batch['video_mask'].cuda()

        # encode
        hidden_v, video_outputs = \
            self._model.encoder.forward(var_x,
                                        batch['video_len'],
                                        batch['video_mask'])
        hidden_v = [hidden_v]
        hidden_w = [None]
        hidden_gate = [None]

        batch_size = batch['x'].shape[0]

        # batch_size x beam_size
        scores = torch.zeros(batch_size, 1)
        # seq_length x batch_size x beam_size
        var_predicts = Variable(torch.ones(1, batch_size, 1).long() * 2)
        # batch_size x beam_size
        if_end = torch.zeros(batch_size, 1).byte()

        if self._use_cuda:
            scores = scores.cuda()
            var_predicts = var_predicts.cuda()
            if_end = if_end.cuda()

        predict_len = 0
        while not if_end.all() \
                and predict_len < max_len:
            beam_scores = []
            beam_hidden_v, beam_hidden_w, beam_hidden_gate = [], [], []
            for i in range(var_predicts.data.shape[-1]):
                # take previous label according to if_teach
                prev_word = var_predicts[-1, :, i]

                # batch_size x
                prev_word = prev_word.detach()

                # decode
                step_logits, step_hidden_v, step_hidden_w, step_hidden_gate = \
                    self._model.decoder.forward(
                        video_outputs, batch['video_mask'],
                        prev_word,
                        hidden_v[i], hidden_w[i], hidden_gate[i])

                # beam_size x [batch_size x word_dim]
                beam_scores.append(
                    torch.nn.functional.log_softmax(step_logits).data
                    + (scores[:, i] * (~if_end[:, i]).float())
                    .unsqueeze(-1))
                # beam_size x [batch_size x hidden]
                beam_hidden_v.append(step_hidden_v)
                # beam_size x [batch_size x hidden]
                beam_hidden_w.append(step_hidden_w)
                # beam_size x [batch_size x hidden]
                beam_hidden_gate.append(step_hidden_gate)

            # batch_size x (beam_size x word_dim)
            beam_scores = torch.cat(beam_scores, -1)

            # get top k
            scores, best_indices = torch.topk(beam_scores,
                                              beam_size,
                                              -1, sorted=False)

            # batch_size x beam_size
            best_beam_indices = best_indices / self._word_dim

            # beam_size x batch_size x hidden
            beam_hidden_v = \
                (torch.cat([hidden[0] for hidden in beam_hidden_v], 0),
                 torch.cat([hidden[1] for hidden in beam_hidden_v], 0))
            beam_hidden_w = \
                (torch.cat([hidden[0] for hidden in beam_hidden_w], 0),
                 torch.cat([hidden[1] for hidden in beam_hidden_w], 0))
            beam_hidden_gate = \
                (torch.cat([hidden[0] for hidden in beam_hidden_gate], 0),
                 torch.cat([hidden[1] for hidden in beam_hidden_gate], 0))

            # Update states with topk beams
            # beam_size x batch_size x hidden
            hidden_v = \
                [(beam_hidden_v[0][
                    best_beam_indices[:, beam],
                    list(range(batch_size))
                ].unsqueeze(0),
                    beam_hidden_v[1][
                    best_beam_indices[:, beam],
                    list(range(batch_size))
                ].unsqueeze(0))
                    for beam in range(beam_size)]
            hidden_w = \
                [(beam_hidden_w[0][
                    best_beam_indices[:, beam],
                    list(range(batch_size))
                ].unsqueeze(0),
                    beam_hidden_w[1][
                    best_beam_indices[:, beam],
                    list(range(batch_size))
                ].unsqueeze(0))
                    for beam in range(beam_size)]
            hidden_gate = \
                [(beam_hidden_gate[0][
                    best_beam_indices[:, beam],
                    list(range(batch_size))
                ].unsqueeze(0),
                    beam_hidden_gate[1][
                    best_beam_indices[:, beam],
                    list(range(batch_size))
                ].unsqueeze(0))
                    for beam in range(beam_size)]
            var_predicts = [var_predicts[:,
                                         list(range(batch_size)),
                                         best_beam_indices[:, beam]]
                            .unsqueeze(-1)
                            for beam in range(beam_size)]
            # seq_length x batch_size x beam_size
            var_predicts = torch.cat(
                var_predicts,
                -1)

            # batch_size x beam_size
            chosen_words = best_indices % self._word_dim

            # Append chosen words prediction
            var_predicts = \
                torch.cat([var_predicts,
                           chosen_words.unsqueeze(0)],
                          0)

            if_end = torch.sum(var_predicts == 1, 0).data
            predict_len += 1

        _, best_score_indices = torch.max(scores, -1)
        var_predicts = var_predicts[:,
                                    list(range(batch_size)),
                                    best_score_indices]

        return var_predicts.data.cpu().numpy().T