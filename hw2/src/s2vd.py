import torch
from torch.autograd import Variable


class S2VD(torch.nn.Module):
    def __init__(self, frame_dim, word_dim, hidden_size=256):
        super(S2VD, self).__init__()

        self.gru_video = torch.nn.GRU(frame_dim,
                                      hidden_size,
                                      1,
                                      bidirectional=False)
        self.gru_caption = torch.nn.GRU(hidden_size + word_dim,
                                        hidden_size,
                                        1,
                                        bidirectional=False)
        self.linear_out = torch.nn.Linear(hidden_size, word_dim)
        self._frame_dim = frame_dim
        self._word_dim = word_dim
        self._hidden_size = hidden_size

        class_weights = torch.ones(word_dim)
        class_weights[0] = 0

    def forward(self, loss_func, batch, training=True, epoch=0):
        batch_size = batch['x'].shape[0]

        # calc video lengths
        video_lengths = torch.sum(torch.sum(batch['x'], dim=-1) > 0,
                                  dim=-1).tolist()

        batch['x'] = Variable(batch['x'])
        # encode video
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            batch['x'].transpose(0, 1), video_lengths)
        hidden_video = None
        outputs, hidden_video = self.gru_video(packed, hidden_video)

        # add padding for text
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        padding = torch.zeros(outputs.data.shape[0],
                              outputs.data.shape[1],
                              self._word_dim)
        padding = Variable(padding)
        outputs = torch.cat([outputs, padding], dim=-1)

        # prepare for caption
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            outputs, video_lengths)
        hidden_caption = None
        outputs, hidden_video = self.gru_caption(packed, hidden_caption)

        # init prev_pred with <sos>
        probs = torch.zeros(batch_size,
                            max(batch['lengths']),
                            self._word_dim)
        probs[0, :, 1] = 1
        probs = Variable(probs)

        loss = Variable(torch.zeros(1))

        # make padding padding for video
        padding = Variable(torch.zeros(1, batch_size,
                                       self._frame_dim))
        for i in range(1, max(batch['lengths'])):
            outputs, hidden_video = self.gru_video(padding, hidden_video)

            # convert label to one-hot
            prev_label = torch.zeros(1, batch_size, self._word_dim)
            prev_label.scatter_(2, batch['y'][i - 1].view(1, -1, 1),
                                torch.ones(1, batch_size, 1))
            prev_label = Variable(prev_label)

            # predict word
            outputs, hidden_caption = \
                self.gru_caption(torch.cat([outputs, prev_label], dim=-1),
                                 hidden_video)
            probs[:, i] = self.linear_out(outputs)
            loss = loss_func(probs[:, i], Variable(batch['y'][i]))

        # loss /= sum(batch['lengths'])

        return probs, loss


# def loss(predicts, labels):
    
