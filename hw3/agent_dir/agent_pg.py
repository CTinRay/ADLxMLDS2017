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
        self._max_iters = args.max_timesteps
        self._iter = 1

        self._model = Policy(self.env.observation_space.shape,
                             self.env.action_space.n)
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            self._model = self._model.cuda()
            torch.cuda.manual_seed_all(0)
        self._optimizer = torch.optim.RMSprop(self._model.parameters(),
                                              lr=1e-4)

        if args.test_pg or args.model:
            print('loading trained model')
            if self._use_cuda:
                ckp = torch.load(args.model_pg)
            else:
                ckp = torch.load(args.model_pg,
                                 map_location=lambda storage, loc: storage)

            self._model.load_state_dict(ckp['model'])

        self._prev_obs = 0

    def init_game_setting(self):
        pass

    def _preprocess_obs(self, obs):
        obs = obs[34:194]
        # obs = obs[::2, ::2, :1]
        # obs = obs[:, :, 0]
        obs[obs[:, :, 0] == 144] = 0
        obs[obs[:, :, 0] == 109] = 0
        # obs[obs != 0] = 1
        # processed = obs - self._prev_obs
        obs = 0.2126 * obs[:, :, 0] \
              + 0.7152 * obs[:, :, 1] \
              + 0.0722 * obs[:, :, 2]
        obs = obs.astype(np.uint8)
        obs = scipy.misc.imresize(obs, (80, 80)).astype(float)
        processed = obs - self._prev_obs
        self._prev_obs = obs
        return processed

    def _train_iteration(self, mean):
        obs = self.env.reset()
        n_steps = 0
        done = False
        rewards = [0]
        self._optimizer.zero_grad()
        while not done:
            total_log_probs = 0
            total_entropy = 0
            while rewards[-1] == 0:
                # calculate action probability
                var_obs = Variable(torch.from_numpy(self._preprocess_obs(obs))
                                   .float().unsqueeze(0))
                if self._use_cuda:
                    var_obs = var_obs.cuda()
                action_probs = self._model.forward(var_obs)
                entropy = - (action_probs * action_probs.log()).sum()
                total_entropy += entropy

                # sample action
                action = torch.multinomial(action_probs, 1).data[0, 0]

                obs, reward, done, _ = self.env.step(action)

                # accumulate reward and probability
                rewards[-1] += reward
                total_log_probs += action_probs[:, action].log()

                n_steps += 1

            loss = -(reward - mean) * total_log_probs
            loss.backward()
            rewards.append(0)

        torch.nn.utils.clip_grad_norm(self._model.parameters(),
                                      5, 'inf')
        self._optimizer.step()
        self._optimizer.zero_grad()

        return rewards, n_steps

    def train(self):
        if self.log_file is not None:
            fp_log = open(self.log_file, 'w', buffering=1)

        total_steps = 0
        rewards = [-1.0]
        while self._iter < self._max_iters:
            mean = sum(rewards[-2000:]) / len(rewards[-2000:])
            reward, n_steps = self._train_iteration(mean)
            total_steps += n_steps
            rewards += reward

            if self.log_file is not None:
                fp_log.write('{},{}\n'.format(total_steps, sum(reward)))

            print(sum(reward))
            if self._iter % 10 == 0:
                print('%d %d %f' % (self._iter, n_steps, mean))

            if self._iter % 100 == 0:
                torch.save({'model': self._model.state_dict(),
                            'iter': self._iter}, 'model-pg-pong')

            self._iter += 1

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
        action_probs = self._model.forward(var_obs)

        # sample action
        action = torch.multinomial(action_probs, 1).data[0, 0]

        return action


class Policy(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Policy, self).__init__()
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

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2048, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions),
            torch.nn.Softmax()
        )

    def forward(self, frames):
        frames = frames.unsqueeze(-3)
        x = self.cnn(frames)
        x = x.view(x.size(0), -1)
        return self.mlp(x)
