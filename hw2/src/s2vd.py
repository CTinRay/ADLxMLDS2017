import torch
import numpy as np
from torch.autograd import Variable
import pdb


use_cuda = torch.cuda.is_available()


class S2VD(torch.nn.Module):
    def __init__(self, frame_dim, word_dim, hidden_size=1024,
                 embedding_dim=500):
        super(S2VD, self).__init__()

        self.lstm_video = torch.nn.LSTM(frame_dim,
                                        hidden_size,
                                        1,
                                        bidirectional=False)
        self.lstm_caption = torch.nn.LSTM(hidden_size + embedding_dim,
                                          hidden_size,
                                          1,
                                          bidirectional=False)
        self.linear_out = torch.nn.Linear(hidden_size, word_dim)
        self.embedding = torch.nn.Embedding(word_dim, embedding_dim)
        self._frame_dim = frame_dim
        self._word_dim = word_dim
        self._hidden_size = hidden_size
        self._embedding_dim = embedding_dim

        class_weights = torch.ones(word_dim)
        class_weights[0] = 0

    def forward(self, loss_func, batch, training=True, epoch=0):
        batch_size = batch['x'].shape[0]

        # calc video lengths
        video_lengths = torch.sum(torch.sum(batch['x'], dim=-1) > 0,
                                  dim=-1).tolist()

        batch['x'] = Variable(batch['x'].cuda())
        # encode video
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            batch['x'].transpose(0, 1), video_lengths)
        hidden_video = None
        outputs, hidden_video = self.lstm_video(packed, hidden_video)

        # add padding for text
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        padding = torch.zeros(outputs.data.shape[0],
                              outputs.data.shape[1],
                              self._embedding_dim)
        padding = Variable(padding.cuda())
        outputs = torch.cat([outputs, padding], dim=-1)

        # prepare for caption
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            outputs, video_lengths)
        hidden_caption = None
        outputs, hidden_video = self.lstm_caption(packed, hidden_caption)

        # init prev_pred with <sos>
        predicts = torch.zeros(batch_size,
                               max(batch['lengths']))
        predicts[:, 0] = 2
        predicts.cuda()

        loss = Variable(torch.zeros(1).cuda())

        # make padding padding for video
        padding = Variable(torch.zeros(1, batch_size,
                                       self._frame_dim).cuda())
        for i in range(1, max(batch['lengths'])):
            outputs, hidden_video = self.lstm_video(padding, hidden_video)

            # flip coins with teach_prob
            teach_prob = 1 - (epoch / (150 + epoch)) if training else 0
            if_teach = np.random.binomial(1, teach_prob, batch_size)
            if_teach = torch.from_numpy(if_teach)

            # take previous label according to if_teach
            prev_label = batch['y'][i - 1] * if_teach \
                + torch.Tensor.long(predicts[:, i - 1]) * (1 - if_teach)
            # convert label to embedding
            prev_word = self.embedding(
                Variable(prev_label.unsqueeze(0).cuda()))

            # predict word
            outputs, hidden_caption = \
                self.lstm_caption(torch.cat([outputs, prev_word], dim=-1),
                                  hidden_video)
            prob = self.linear_out(outputs)[0]
            loss = loss + loss_func(prob, Variable(batch['y'][i].cuda()))
            _, predicts[:, i] = torch.max(prob.data, -1)

        loss /= sum(batch['lengths'])

        return predicts, loss
