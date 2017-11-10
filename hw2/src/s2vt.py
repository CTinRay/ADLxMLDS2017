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
                training):
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
                             hidden_caption)
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
