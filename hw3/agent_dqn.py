import pdb
import random
import numpy as np
import torch
from torch.autograd import Variable
from replay_buffer import ReplayBuffer
from q import Q


class AgentDQN:
    def __init__(self, env, args):
        self.t = 0
        self.env = env
        self.n_actions = self.env.action_space.n
        self.max_timesteps = args['max_timesteps']
        self.gamma = args['gamma']
        self.exploration_final_eps = args['exploration_final_eps']
        self.batch_size = args['batch_size']
        self.prioritized_replay_eps = args['prioritized_replay_eps']
        self.target_network_update_freq = args['target_network_update_freq']
        self.replay_buffer = ReplayBuffer(args['buffer_size'])

        self._model = Q(self.env.observation_space.shape,
                        self.env.action_space.n)
        self._use_cuda = torch.cuda.is_available()
        self._loss = torch.nn.SmoothL1Loss()
        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=args['learning_rate'])
        if self._use_cuda:
            self._model = self._model.cuda()

    def init_game_setting(self):
        pass

    def make_action(self, state, test):

        # decide if doing exploration
        if not test:
            epsilon = 1 \
                - (1 - self.exploration_final_eps) \
                * self.t / 100000
            epsilon = max(epsilon, 0)
            explore = random.random() < epsilon
        else:
            explore = False

        if explore:
            return random.randint(0, self.n_actions - 1)
        else:
            state = torch.from_numpy(state).float()
            state = Variable(state, volatile=True)
            if self._use_cuda:
                state = state.cuda()
            action_value = self._model.forward(state.unsqueeze(0))
            best_action = action_value.max(-1)[1].data.cpu().numpy()
            return best_action[0]

    def update_model(self, target_q):
        # sample from replay_buffer
        replay = self.replay_buffer.sample(self.batch_size, beta=1)

        # prepare tensors
        tensor_replay = [torch.from_numpy(val) for val in replay]
        if self._use_cuda:
            tensor_replay = [val.cuda() for val in tensor_replay]
        states0, actions, rewards, states1, dones, \
            _, weights = tensor_replay

        # predict target with target network
        var_states1 = Variable(states1.float())
        var_target_reward = \
            target_q.forward(var_states1).max(-1)[0]
        var_targets = Variable(rewards) \
            + self.gamma * var_target_reward * (-Variable(dones) + 1)
        var_targets = var_targets.detach()

        # gradient descend model
        var_states0 = Variable(states0.float())
        var_action_values = self._model.forward(var_states0)\
            .gather(1, Variable(actions.view(-1, 1)))
        var_loss = self._loss(var_action_values, var_targets)

        # weighted sum loss
        var_weights = Variable(weights)
        var_loss_sum = torch.sum(var_loss * var_weights)

        # gradient descend loss
        self._optimizer.zero_grad()
        var_loss_sum.backward()
        self._optimizer.step()

        # update experience priorities
        indices = replay[5]
        loss = var_loss.data.cpu().numpy()
        new_priority = loss + self.prioritized_replay_eps
        self.replay_buffer.update_priorities(indices, new_priority)

        return np.mean(loss)

    def train(self):
        target_q = Q(self.env.observation_space.shape,
                     self.env.action_space.n)
        if self._use_cuda:
            target_q = target_q.cuda()
        target_q.load_state_dict(self._model.state_dict())

        state0 = self.env.reset()

        # log statics
        episode_rewards = [0]
        for self.t in range(self.max_timesteps):
            # play
            action = self.make_action(np.array(state0), False)
            state1, reward, done, _ = self.env.step(action)
            self.replay_buffer.add(state0, action,
                                   reward, state1, done)
            # accumulate episode reward
            episode_rewards[-1] += reward

            # update previous state
            if done:
                state0 = self.env.reset()
                print('t = %d, r = %f' % (self.t, episode_rewards[-1]))
                episode_rewards.append(0)
            else:
                state0 = state1

            # train on batch
            loss = self.update_model(target_q)

            # update target network
            if self.t % self.target_network_update_freq == 0:
                target_q.load_state_dict(self._model.state_dict())
