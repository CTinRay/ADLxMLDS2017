import torch
import numpy as np
from torch.autograd import Variable
import pdb


class HLSTMatEncoder(torch.nn.Module):
    def __init__(self, frame_dim, hidden_dim):
        super(HLSTMatEncoder, self).__init__()
        self.lh = torch.nn.Linear(frame_dim, hidden_dim, False)
        self.lc = torch.nn.Linear(frame_dim, hidden_dim, False)
        self.tanh = torch.nn.Tanh()

    def forward(self, frames, video_mask):
        """ Forward

        args:
            frame (Variable): (time, batch, feature)
            video_mask (Variable): (time, batch, feature)
        """
        frame_sum = torch.sum(frames * video_mask, dim=0)
        frame_mean = frame_sum / torch.sum(video_mask, dim=0)

        h0 = self.tanh(self.lh(frame_mean).unsqueeze(0))
        c0 = self.tanh(self.lc(frame_mean).unsqueeze(0))
        return None, (h0, c0)


class RelevantScore(torch.nn.Module):
    def __init__(self, dim1, dim2, hidden1):
        super(RelevantScore, self).__init__()
        self.lW1 = torch.nn.Linear(dim1, hidden1, bias=False)
        self.lW2 = torch.nn.Linear(dim2, hidden1, bias=False)
        self.b = torch.nn.Parameter(torch.Tensor(hidden1))
        self.tanh = torch.nn.Tanh()
        self.lw = torch.nn.Linear(hidden1, 1, bias=False)

    def forward(self, input1, input2):
        return self.lw(self.tanh(self.lW1(input1) + self.lW2(input2) + self.b))


class HLSTMatDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, frame_dim, embed_dim, word_dim):
        super(HLSTMatDecoder, self).__init__()
        self.rnn1 = torch.nn.LSTM(embed_dim,
                                  hidden_dim,
                                  1,
                                  bidirectional=False)
        self.rnn2 = torch.nn.LSTM(hidden_dim,
                                  hidden_dim,
                                  1,
                                  bidirectional=False)
        self.embedding = torch.nn.Embedding(word_dim, embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + frame_dim, embed_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(embed_dim, word_dim))
        self.relevant_score = RelevantScore(frame_dim, hidden_dim, 128)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, var_x, video_mask,
                prev_word,
                hidden1, hidden2):
        """ Forward

        args:
            var_x (Variable): (time, batch_size, feature)
            video_mask (Variable): (time, batch_size, feature)
            prev_word (Variable): (batch_size, 1)
            hidden1 (Variable): (batch_size, h_dim)
            hidden2 (Variable): (batch_size, h_dim)
            training (bool): If training.
        """

        # convert label to embedding
        prev_word = self.embedding(prev_word)

        # feed rnns
        outputs2, hidden2 = self.rnn2(prev_word.unsqueeze(0), hidden2)
        outputs2 = self.dropout(outputs2)
        outputs1, hidden1 = self.rnn1(outputs2, hidden1)
        outputs1 = self.dropout(outputs1)

        # time x batch x 1
        relevant_scores = self.relevant_score(var_x, outputs2)
        e_relevant_scores = torch.exp(relevant_scores) * video_mask
        weights = e_relevant_scores / torch.sum(e_relevant_scores, 0)
        attention = torch.sum(weights * var_x, 0)
        attention = self.dropout(attention)

        logits = self.mlp(torch.cat([outputs1.squeeze(0), attention], -1))
        return logits, hidden1, hidden2


class HLSTMat(torch.nn.Module):
    def __init__(self, frame_dim, word_dim,
                 hidden_dim=512, embed_dim=512):
        super(HLSTMat, self).__init__()
        self.encoder = HLSTMatEncoder(frame_dim, hidden_dim)
        self.decoder = HLSTMatDecoder(hidden_dim, frame_dim,
                                      embed_dim, word_dim)
