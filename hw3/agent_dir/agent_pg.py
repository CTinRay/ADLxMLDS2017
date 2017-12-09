import math
import pdb
import torch
from torch.autograd import Variable
import scipy.misc
import numpy as np


class Agent_PG():
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        self.env = env
        self.batch_size = args.batch_size
        self.log_file = args.log_file
        self._max_timesteps = args.max_timesteps
        self._n_steps = 0

        self._model = PolicyValueNet(self.env.observation_space.shape,
                                     self.env.action_space.n)
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            self._model = self._model.cuda()
        self._optimizer = torch.optim.RMSprop(self._model.parameters(),
                                              lr=1e-4)
        self._value_coef = 0.5

        if args.test_pg or args.model:
            print('loading trained model')
            if self._use_cuda:
                ckp = torch.load(args.test_pg)
            else:
                ckp = torch.load(args.test_pg,
                                 map_location=lambda storage, loc: storage)

            self._model.load_state_dict(ckp['model'])

        self._prev_obs = 0

    def init_game_setting(self):
        pass

    def _preprocess_obs(self, obs):
        obs = obs[34:194]
        obs[obs[:, :, 0] == 144] = 0
        obs[obs[:, :, 0] == 109] = 0
        obs = 0.2126 * obs[:, :, 0] \
            + 0.7152 * obs[:, :, 1] \
            + 0.0722 * obs[:, :, 2]
        obs = scipy.misc.imresize(obs, (80, 80)).astype(float)
        processed = obs - self._prev_obs
        self._prev_obs = obs
        return processed

    def _approx_return(self, rewards, next_state):
        """Note that it is specialized for games that has only
           one none-zero rewards for each episode.
        """
        returns = list(rewards)
        if returns[-1] == 0:
            var_next_state = \
                Variable(torch.from_numpy(next_state).float()) \
                .unsqueeze(0)
            if self._use_cuda:
                var_next_state = var_next_state.cuda()
            _, var_value = self._model.forward(var_next_state)
            returns.append(var_value.data.cpu()[0, 0])

        # TODO: consider gamma
        for i in range(-2, -len(returns) - 1, -1):
            if returns[i] == 0:
                returns[i] = returns[i + 1]

        return returns[:len(rewards)]

    def train(self):
        if self.log_file is not None:
            fp_log = open(self.log_file, 'w', buffering=1)

        # used to print
        rewards = [0]
        best_mean_reward = -21

        # used to update policy
        var_batch_action_probs = []
        var_batch_values = []
        var_batch_entropy = 0
        batch_rewards = []

        # used to update value
        var_episode_values = []

        obs = self._preprocess_obs(self.env.reset())
        while self._n_steps < self._max_timesteps:
            # convert observation to Variable
            var_obs = Variable(torch.from_numpy(obs).float())
            if self._use_cuda:
                var_obs = var_obs.cuda()

            # make action
            action_probs, value = self._model.forward(var_obs.unsqueeze(0))
            action = torch.multinomial(action_probs, 1).data[0, 0]
            entropy = torch.sum(action_probs * torch.log(action_probs))
            obs, reward, done, _ = self.env.step(action)
            obs = self._preprocess_obs(obs)

            # save reward, action_probs and value
            var_batch_action_probs.append(action_probs[0, action])
            var_batch_entropy += entropy
            var_batch_values.append(value[0, 0])
            batch_rewards.append(reward)
            var_episode_values.append(value)

            # update policy
            if len(var_batch_values) == self.batch_size:
                var_action_probs = torch.cat(var_batch_action_probs)
                var_values = torch.cat(var_batch_values).detach()
                returns = self._approx_return(batch_rewards, obs)
                var_returns = Variable(torch.Tensor(returns))
                if self._use_cuda:
                    var_returns = var_returns.cuda()

                loss = - torch.mean(
                    torch.log(var_action_probs) * (var_returns - var_values)) \
                    - 1e-3 * var_batch_entropy / self.batch_size

                # update model
                self._optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(self._model.parameters(),
                                              5, 'inf')
                self._optimizer.step()

                var_batch_action_probs = []
                var_batch_values = []
                var_batch_entropy = 0
                batch_rewards = []

            # update value
            if reward != 0:
                var_values = torch.cat(var_episode_values)

                # TODO: Consider gamma when calculate loss
                loss = self._value_coef * torch.mean((var_values - reward)**2)

                # update model
                self._optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self._optimizer.step()

                var_episode_values = []

            # print and log
            rewards[-1] += reward
            if done:
                obs = self._preprocess_obs(self.env.reset())
                if self.log_file is not None:
                    fp_log.write('{},{}\n'.format(self._n_steps, rewards[-1]))

                print('{} {}'
                      .format(self._n_steps, rewards[-1]))

                mean_reward = sum(rewards[-100:]) / len(rewards[-100:])
                # if mean_reward > best_mean_reward:
                if True:
                    best_mean_reward = mean_reward
                    torch.save({'model': self._model.state_dict()},
                               'model-pg-pong')

                rewards.append(0)

            self._n_steps += 1

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        var_obs = Variable(torch.from_numpy(self._preprocess_obs(observation))
                           .float().unsqueeze(0))
        if self._use_cuda:
            var_obs = var_obs.cuda()
        action_probs, _ = self._model.forward(var_obs)

        # sample action
        action = torch.multinomial(action_probs, 1).data[0, 0]

        return action


class PolicyValueNet(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyValueNet, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 4, stride=2),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),
            # torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2)
        )

        # init weights wit Lecun Normal
        # for m in self.cnn.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #         m.weight.data.normal_(0, math.sqrt(1. / n))

        self.mlp_action = torch.nn.Sequential(
            torch.nn.Linear(2048, n_actions),
            torch.nn.Softmax()
        )
        self.mlp_advantage = torch.nn.Sequential(
            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, frames):
        frames = frames.unsqueeze(-3)
        x = self.cnn(frames)
        x = x.view(x.size(0), -1)
        return self.mlp_action(x), self.mlp_advantage(x)
