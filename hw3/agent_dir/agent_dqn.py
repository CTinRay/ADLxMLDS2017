import pdb
import random
import numpy as np
import torch
from torch.autograd import Variable
from utils.openai_replay_buffer import PrioritizedReplayBuffer as ReplayBuffer


class Agent_DQN:
    def __init__(self, env, args):
        self.t = 0
        self.env = env
        self.n_actions = self.env.action_space.n
        self.max_timesteps = args.max_timesteps
        self.gamma = args.gamma
        self.exploration_final_eps = args.exploration_final_eps
        self.exploration_steps = args.exploration_steps
        self.batch_size = args.batch_size
        self.prioritized_replay_eps = args.prioritized_replay_eps
        self.target_network_update_freq = args.target_network_update_freq
        self.replay_buffer = ReplayBuffer(int(args.buffer_size), 0.6)
        self.save_freq = args.save_freq
        self.log_file = args.log_file

        self._model = Q(self.env.observation_space.shape,
                        self.env.action_space.n)
        self._use_cuda = torch.cuda.is_available()
        # self._loss = torch.nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=args.learning_rate)
        if self._use_cuda:
            self._model = self._model.cuda()

        if args.test_dqn:
            print('loading trained model')
            if self._use_cuda:
                ckp = torch.load(args.test_dqn)
            else:
                ckp = torch.load(args.test_dqn,
                                 map_location=lambda storage, loc: storage)

            self._model.load_state_dict(ckp['model'])

    def init_game_setting(self):
        pass

    def make_action(self, state, test):

        # decide if doing exploration
        if not test:
            epsilon1 = 1 - (1 - 0.1) * self.t / (self.max_timesteps / 500)
            epsilon2 = 0.1 - (0.1 - 0.01) * self.t / (self.max_timesteps / 50)
            self.epsilon = max(epsilon1, epsilon2, 0.01)
        else:
            self.epsilon = 0.01
        explore = random.random() < self.epsilon

        if explore:
            return random.randint(0, self.n_actions - 1)
        else:
            state = np.array(state)
            state = torch.from_numpy(state).float()
            if self._use_cuda:
                state = state.cuda()
            action_value = self._model.forward(Variable(state.unsqueeze(0)))
            best_action = action_value.data.max(-1)[1].cpu().numpy()
            return best_action[0]

    def update_model(self, target_q):
        # sample from replay_buffer
        beta = 0.4 \
            + (1.0 - 0.40) \
            * self.t / self.max_timesteps
        beta = min(beta, 1)
        replay = self.replay_buffer.sample(self.batch_size, beta=beta)

        # prepare tensors
        tensor_replay = [torch.from_numpy(val) for val in replay[:-1]]
        if self._use_cuda:
            tensor_replay = [val.cuda() for val in tensor_replay]
        states0, actions, rewards, states1, dones, weights = tensor_replay

        # predict target with target network
        var_target_action_value = \
            target_q.forward(Variable(states1.float())).max(-1)[0]
        var_target = Variable(rewards.float()) \
            + self.gamma * var_target_action_value \
            * (Variable(-dones.float() + 1))
        var_target = var_target.unsqueeze(-1)

        # gradient descend model
        var_states0 = Variable(states0.float())
        var_action_value = self._model.forward(var_states0) \
            .gather(1, Variable(actions.view(-1, 1)))
        var_loss = (var_action_value - var_target) ** 2 / self.batch_size

        if self.t % 5000 == 0:
            print(var_target)

        # weighted sum loss
        var_weights = Variable(weights.float())
        var_loss_mean = torch.mean(var_loss.squeeze(-1) * var_weights)

        # gradient descend loss
        self._optimizer.zero_grad()
        var_loss_mean.backward()
        self._optimizer.step()

        # update experience priorities
        indices = replay[-1]
        loss = torch.abs(var_action_value - var_target).data
        new_priority = loss + self.prioritized_replay_eps
        new_priority = new_priority.view(-1,).cpu().tolist()
        self.replay_buffer.update_priorities(indices, new_priority)

        return np.mean(loss.cpu().numpy())

    def train(self):
        # init target network
        target_q = Q(self.env.observation_space.shape,
                     self.env.action_space.n)
        if self._use_cuda:
            target_q = target_q.cuda()
        target_q.load_state_dict(self._model.state_dict())

        state0 = self.env.reset()

        # log statics
        loss = 0
        episode_rewards = [0]
        best_mean_reward = 0

        if self.log_file is not None:
            fp_log = open(self.log_file, 'w', buffering=1)

        while self.t < self.max_timesteps:
            # play
            for i in range(4):
                action = self.make_action(state0, False)
                state1, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(state0, action,
                                       float(reward), state1, float(done))
                # accumulate episode reward
                episode_rewards[-1] += reward

                # update previous state and log
                if done:
                    state0 = self.env.reset()
                    print('t = %d, r = %f, loss = %f, exp = %f'
                          % (self.t, episode_rewards[-1], loss, self.epsilon))
                    if self.log_file is not None:
                        fp_log.write('{},{},{}\n'.format(self.t,
                                                         episode_rewards[-1],
                                                         loss))
                    episode_rewards.append(0)
                else:
                    state0 = state1

            if self.t * 4 > self.replay_buffer._maxsize:
                # train on batch
                loss = self.update_model(target_q)

            # update target network
            if self.t % self.target_network_update_freq == 0:
                target_q.load_state_dict(self._model.state_dict())

            if self.t % self.save_freq == 0:
                mean_reward = \
                    sum(episode_rewards[-100:]) / len(episode_rewards[-100:])
                if best_mean_reward < mean_reward:
                    print('save best model with mean reward = %f'
                          % mean_reward)
                    best_mean_reward = mean_reward
                    torch.save({
                        'model': self._model.state_dict()
                    }, 'model')

            self.t += 1


class Q(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Q, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[-1], 32, 8, stride=4),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 3, stride=2),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2)
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
        )

    def forward(self, frames):
        frames = frames.transpose(-3, -1)
        x = self.cnn(frames)
        x = x.view(x.size(0), -1)
        return self.mlp(x)
