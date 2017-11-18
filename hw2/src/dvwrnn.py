import pdb
import numpy as np
import torch
from torch.autograd import Variable


class DVWRNNEncoder(torch.nn.Module):
    def __init__(self, frame_dim, hidden_dim, conv_dim=1024):
        super(DVWRNNEncoder, self).__init__()
        # self.rnn = torch.nn.LSTM(frame_dim,
        #                          hidden_dim,
        #                          1,
        #                          bidirectional=False,
        #                          dropout=0)
        # self.cnn = torch.nn.Conv1d(frame_dim,
        #                            conv_dim,
        #                            5, padding=2)
        self.lh = torch.nn.Linear(frame_dim, hidden_dim, False)
        self.lc = torch.nn.Linear(frame_dim, hidden_dim, False)
        self.tanh = torch.nn.Tanh()

    def forward(self, frames, video_lens, video_mask):
        """ Forward

        args:
            frame (Variable): (time, batch, feature)
            video_mask (Variable): (time, batch, feature)
        """
        # encode video
        # frames = self.cnn(frames.transpose(0, 1).transpose(1, 2))
        # frames = torch.nn.functional.relu(frames)
        # frames = frames.transpose(1, 2).transpose(0, 1)
        frame_sum = torch.sum(frames * video_mask, dim=0)
        frame_mean = frame_sum / torch.sum(video_mask, dim=0)

        h0 = self.tanh(self.lh(frame_mean).unsqueeze(0))
        c0 = self.tanh(self.lc(frame_mean).unsqueeze(0))
        return (h0, c0), frames

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            frames, video_lens)
        hidden_video = None
        video_outputs, hidden_video = self.rnn(packed, hidden_video)

        # unpack for padding
        video_outputs, _ = \
            torch.nn.utils.rnn.pad_packed_sequence(video_outputs)

        return hidden_video, video_outputs


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


class DVWRNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, frame_dim, embed_dim, word_dim,
                 gate_hidden_dim=128):
        super(DVWRNNDecoder, self).__init__()
        self.rnn_v = torch.nn.LSTM(embed_dim,
                                   embed_dim,
                                   1,
                                   bidirectional=False,
                                   dropout=0)
        self.rnn_w = torch.nn.LSTM(embed_dim,
                                   embed_dim,
                                   1,
                                   bidirectional=False,
                                   dropout=0)
        self.rnn_gate = torch.nn.LSTM(embed_dim,
                                      embed_dim,
                                      1,
                                      bidirectional=False,
                                      dropout=0)
        self.embedding = torch.nn.Embedding(word_dim,
                                            embed_dim)
        self.dropout = torch.nn.Dropout(0.5)

        # attention score
        self.relevant_score = RelevantScore(frame_dim, embed_dim, 128)

        # for gate
        self.mlp_gate = torch.nn.Sequential(
            # torch.nn.Linear(gate_hidden_dim, 1),
            torch.nn.Sigmoid())
        self.mlp_attention = torch.nn.Sequential(
            torch.nn.Linear(frame_dim, embed_dim),
            torch.nn.Tanh())
        self.mlp_out = torch.nn.Sequential(
            torch.nn.Linear(2 * embed_dim, embed_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(embed_dim, word_dim)
        )

    def forward(self, video_outputs, video_mask,
                prev_word,
                hidden_v, hidden_w, hidden_gate):
        """ Forward

        args:
            video_outputs (Variable): (time, batch_size, feature)
            video_mask (Variable): (time, batch_size, feature)
            prev_word (Variable): (batch_size, 1)
            hidden_v (Variable): (batch_size, embed_dim)
            hidden_w (Variable): (batch_size, embed_dim)
            hidden_gate (Variable): (batch_size, gate_hidden_dim)
        """

        # convert label to embedding
        prev_word = self.embedding(prev_word)
        prev_word = self.dropout(prev_word)

        # feed rnns
        output_w, hidden_w = self.rnn_w(prev_word.unsqueeze(0), hidden_w)
        output_w = self.dropout(output_w)
        output_v, hidden_v = self.rnn_v(output_w, hidden_v)
        output_v = self.dropout(output_v)

        # time x batch x 1
        relevant_scores = self.relevant_score(video_outputs, output_v)
        e_relevant_scores = torch.exp(relevant_scores) * video_mask
        weights = e_relevant_scores / torch.sum(e_relevant_scores, 0)
        attention = torch.sum(weights * video_outputs, 0)
        attention = self.mlp_attention(attention)
        attention = self.dropout(attention)
        # attention = attention.unsqueeze(0)

        # Gate: batch x 2
        gate, hidden_gate = self.rnn_gate(
            prev_word.unsqueeze(0),
            hidden_gate)
        gate = (gate / 0.761594156 + 1) / 2
        # logits = attention * gate + output_v * (1 - gate)
        # logits = logits.squeeze(0)
        output_w = output_w.squeeze(0)
        gate = gate.squeeze(0)
        logits = self.mlp_out(torch.cat([attention * gate,
                                         output_w * (1 - gate)], -1))

        return logits, hidden_v, hidden_w, hidden_gate


class DVWRNN(torch.nn.Module):
    def __init__(self, frame_dim, word_dim,
                 hidden_dim=512, embed_dim=512):
        super(DVWRNN, self).__init__()
        self.encoder = DVWRNNEncoder(frame_dim, hidden_dim)
        self.decoder = DVWRNNDecoder(hidden_dim, frame_dim,
                                     embed_dim, word_dim)
