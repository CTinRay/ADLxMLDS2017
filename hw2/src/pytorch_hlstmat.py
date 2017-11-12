import pdb
import torch
from torch.autograd import Variable
from pytorch_base import TorchBase
from hlstmat import HLSTMat


class TorchHLSTMat(TorchBase):
    def __init__(self, frame_dim, word_dim, **kwargs):
        super(TorchHLSTMat, self).__init__(**kwargs)
        self._word_dim = word_dim
        self._model = HLSTMat(frame_dim, word_dim)

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
        hidden1, hidden2 = \
            self._model.encoder.forward(var_x,
                                        batch['video_mask'],
                                        training)

        var_loss = 0
        batch_size = batch['x'].shape[0]

        var_predicts = Variable(torch.ones(1, batch_size).long() * 2)
        var_ones = Variable(torch.ones(batch_size))
        if self._use_cuda:
            var_predicts = var_predicts.cuda()
            var_ones = var_ones.cuda()

        for i in range(1, max(batch['caption_len'])):
            # flip coins with teach_prob
            teach_prob = 1 - (3 * self._epoch / (150 + 3 * self._epoch)) \
                if training else 1
            # teach_prob = 1
            if_teach = torch.bernoulli(teach_prob * var_ones)
            if_teach = if_teach.long()

            # take previous label according to if_teach
            prev_word = var_y[i - 1] * if_teach \
                + var_predicts[-1, :] * (1 + if_teach * -1)

            prev_word = prev_word.detach()
            # decode
            logits, hidden_video, hidden_caption = \
                self._model.decoder.forward(var_x, batch['video_mask'],
                                            prev_word,
                                            hidden1, hidden2,
                                            training)

            probs = torch.nn.functional.softmax(logits)
            # store prediction
            var_predicts = \
                torch.cat([var_predicts,
                           torch.multinomial(probs, 1).view(1, -1)],
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
                                        batch['video_mask'],
                                        False)

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
            probs, hidden_video, hidden_caption = \
                self._model.decoder.forward(var_x, batch['video_mask'],
                                            prev_word,
                                            hidden1, hidden2,
                                            False)

            # store prediction
            var_predicts = \
                torch.cat([var_predicts,
                           torch.max(probs, -1)[1].unsqueeze(0)],
                          0)

            if_end = if_end | (var_predicts[-1] == 1).data
            predict_len += 1

        return var_predicts.data.cpu().numpy().T

    def _beam_search_batch(self, batch, beam_size=5):
        return self._predict_batch(batch)
        var_x = Variable(batch['x'].transpose(0, 1), volatile=True)
        batch['video_len'] = batch['video_len'].tolist()

        if self._use_cuda:
            var_x = var_x.cuda()

        # encode
        hidden_video, hidden_caption = \
            self._model.encoder.forward(var_x,
                                        batch['video_mask'],
                                        False)
        hidden_video = [hidden_video]
        hidden_caption = [hidden_caption]

        batch_size = batch['x'].shape[0]

        # beam_size x batch_size
        var_scores = Variable(torch.zeros(1, batch_size))
        # seq_length x batch_size x beam_size
        var_predicts = Variable(torch.ones(1, batch_size, 1).long() * 2)
        # batch_size x beam_size
        if_end = torch.zeros(batch_size, beam_size).byte()

        if self._use_cuda:
            var_scores = var_scores.cuda()
            var_predicts = var_predicts.cuda()
            if_end = if_end.cuda()

        depth = 0
        while not if_end.all() and depth < 30:
            beam_scores, beam_hidden_video, beam_hidden_caption = [], [], []
            for i in range(var_predicts.data.shape[-1]):
                # take previous label according to if_teach
                prev_word = var_predicts[-1, :, i]

                # batch_size x
                prev_word = prev_word.detach()

                # decode
                step_logits, step_hidden_video, step_hidden_caption = \
                    self._model.decoder.forward(
                        prev_word,
                        hidden_video[i], hidden_caption[i],
                        False)

                # beam_size x [batch_size x word_dim]
                beam_scores.append(torch.nn.functional.log_softmax(step_logits)
                                   + var_scores[:, i].unsqueeze(-1))
                # beam_size x [batch_size x hidden]
                beam_hidden_video.append(step_hidden_video)
                # beam_size x [batch_size x hidden]
                beam_hidden_caption.append(step_hidden_caption)

            # batch_size x (beam_size x word_dim)
            beam_scores = torch.cat(beam_scores, -1)

            # get top k
            var_scores, best_indices = torch.topk(beam_scores,
                                                  beam_size,
                                                  -1, sorted=False)

            # batch_size x beam_size
            best_beam_indices = best_indices / self._word_dim

            # beam_size x batch_size x hidden
            beam_hidden_video = \
                (torch.cat([hidden[0] for hidden in beam_hidden_video], 0),
                 torch.cat([hidden[1] for hidden in beam_hidden_video], 0))
            beam_hidden_caption = \
                (torch.cat([hidden[0] for hidden in beam_hidden_caption], 0),
                 torch.cat([hidden[1] for hidden in beam_hidden_caption], 0))

            # Update states with topk beams
            # beam_size x batch_size x hidden
            hidden_video = \
                [(beam_hidden_video[0][
                    best_beam_indices[:, beam].data,
                    list(range(batch_size))
                ].unsqueeze(0),
                  beam_hidden_video[1][
                    best_beam_indices[:, beam].data,
                    list(range(batch_size))
                  ].unsqueeze(0))
                 for beam in range(beam_size)]
            hidden_caption = \
                [(beam_hidden_caption[0][
                    best_beam_indices[:, beam].data,
                    list(range(batch_size))
                ].unsqueeze(0),
                  beam_hidden_caption[1][
                    best_beam_indices[:, beam].data,
                    list(range(batch_size))
                  ].unsqueeze(0))
                 for beam in range(beam_size)]

            var_predicts = [var_predicts[:,
                                         list(range(batch_size)),
                                         best_beam_indices[:, beam].data]
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

            if_end = if_end | (var_predicts[-1] == 1).data
            depth += 1

        _, best_score_indices = torch.max(var_scores, -1)
        var_predicts = var_predicts[:,
                                    list(range(batch_size)),
                                    best_score_indices.data]

        return var_predicts.data.cpu().numpy().T
