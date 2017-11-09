import torch
import numpy as np
from torch.autograd import Variable
import pdb


use_cuda = torch.cuda.is_available()


class S2VTEncoder(torch.nn.Module):
    def __init__(self, rnn_video, rnn_caption,
                 embed_dim=500):
        super(S2VTEncoder, self).__init__()
        self.rnn_video = rnn_video
        self.rnn_caption = rnn_caption
        self.embed_dim = embed_dim

    def forward(self, frames, video_lens, training):
        """ Forward

        args:
            frame (Variable): (batch, time, feature)
            frame_lens (list): (batch,)
        """
        # encode video
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            frames, video_lens)
        hidden_video = None
        video_outputs, hidden_video = self.rnn_video(packed, hidden_video)

        # unpack for padding
        video_outputs, _ = \
            torch.nn.utils.rnn.pad_packed_sequence(video_outputs)

        # make padding
        padding = torch.zeros(video_outputs.data.shape[0],  # time
                              video_outputs.data.shape[1],  # batch
                              self.embed_dim)               # embedding
        padding = Variable(padding.cuda())

        # pad outputs and pack back
        cated_outputs = torch.cat([video_outputs, padding], dim=-1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            cated_outputs, video_lens)

        # prepare for caption
        hidden_caption = None
        outputs, hidden_caption = self.rnn_caption(packed, hidden_caption)

        return hidden_video, hidden_caption


class S2VTDecoder(torch.nn.Module):
    def __init__(self, rnn_video, rnn_caption,
                 hidden_dim, frame_dim, embed_dim, word_dim):
        super(S2VTDecoder, self).__init__()
        self.rnn_video = rnn_video
        self.rnn_caption = rnn_caption
        self.frame_padding = Variable(torch.zeros(frame_dim)).cuda()
        self.linear_out = torch.nn.Linear(hidden_dim, word_dim)
        self.embedding = torch.nn.Embedding(word_dim, embed_dim)

    def forward(self, prev_word,
                hidden_video, hidden_caption,
                epoch, training):
        """ Forward

        args:
            prev_word (Variable): (batch_size, 1)
            hidden_video (Variable): (batch_size, h_dim)
            hidden_caption (Variable): (batch_size, h_dim)
            epoch (int): Number of epochs until now.
            training (bool): If training.
        """
        batch_size = prev_word.data.shape[0]

        # convert label to embedding
        prev_word = self.embedding(prev_word)

        # feed rnn_video with padding
        padding = torch.stack([self.frame_padding] * batch_size).unsqueeze(0)
        outputs_video, hidden_video = self.rnn_video(padding, hidden_video)

        # predict word
        outputs_caption, hidden_caption = \
            self.rnn_caption(torch.cat([outputs_video, prev_word.unsqueeze(0)],
                                       dim=-1),
                             hidden_video)
        probs = self.linear_out(outputs_caption)[0]
        return probs, hidden_video, hidden_caption


class S2VT(torch.nn.Module):
    def __init__(self, frame_dim, word_dim,
                 hidden_dim=1024, embed_dim=500):
        super(S2VT, self).__init__()

        self.rnn_video = torch.nn.LSTM(frame_dim,
                                       hidden_dim,
                                       1,
                                       bidirectional=False)
        self.rnn_caption = torch.nn.LSTM(hidden_dim + embed_dim,
                                         hidden_dim,
                                         1,
                                         bidirectional=False)

        self.encoder = S2VTEncoder(self.rnn_video, self.rnn_caption, embed_dim)
        self.decoder = S2VTDecoder(self.rnn_video, self.rnn_caption,
                                   hidden_dim, frame_dim, embed_dim, word_dim)


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
