import pdb
import torch
from torch.autograd import Variable
from pytorch_base import TorchBase
from q import Q


class TorchQ(TorchBase):
    def __init__(self, input_shape, n_actions, **kwargs):
        super(TorchQ, self).__init__(**kwargs)
        self._model = Q(input_shape, n_actions)
        if self._use_cuda:
            self._model = self._model.cuda()

        self._loss = torch.nn.SmoothL1Loss()

        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=self._learning_rate)

    def _run_iter(self, batch, training):
        var_states = Variable(batch[0])
        var_actions = Variable(batch[1])
        var_targets = Variable(batch[2])
        var_weights = Variable(batch[3])

        if self._use_cuda:
            var_states = var_states.cuda()
            var_actions = var_actions.cuda()
            var_targets = var_targets.cuda()
            var_weights = var_weights.cuda()

        var_states = var_states.transpose(1, -1)
        var_action_values = self._model.forward(var_states) \
                                       .gather(1, var_actions.view(-1, 1))
        var_loss = self._loss(var_action_values, var_targets) * var_weights
        return var_action_values, var_loss

    def _predict_batch(self, batch):
        var_states = Variable(batch)
        if self._use_cuda:
            var_states = var_states.cuda()
        var_states = var_states.transpose(1, -1)
        return self._model.forward(var_states)
