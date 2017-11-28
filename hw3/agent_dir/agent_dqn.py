import pdb
import random
import numpy as np
import torch
from torch.autograd import Variable
from openai_replay_buffer import PrioritizedReplayBuffer as ReplayBuffer
from q import Q


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
        self.replay_buffer = ReplayBuffer(int(args.buffer_size), 1)

        self._model = Q(self.env.observation_space.shape,
                        self.env.action_space.n)
        self._use_cuda = torch.cuda.is_available()
        self._loss = torch.nn.SmoothL1Loss()
        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=args.learning_rate)
        if self._use_cuda:
            self._model = self._model.cuda()

        if args.test_pg:
            print('loading trained model')
            if self._use_cuda:
                ckp = torch.load(args.test_pg)
            else:
                ckp = torch.load(args.test_dqn,
                                 map_location=lambda storage, loc: storage)

            self._model.load_state_dict(ckp['model'])

    def init_game_setting(self):
        pass

    def make_action(self, state, test):

        # decide if doing exploration
        if not test:
            self.epsilon = 1 \
                - (1 - self.exploration_final_eps) \
                * self.t / self.exploration_steps
            self.epsilon = max(self.epsilon, self.exploration_final_eps)
        else:
            self.epsilon = self.exploration_final_eps
        explore = random.random() < self.epsilon

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
        beta = 0.1 \
            + (1 - 0.10) \
            * self.t / self.max_timesteps
        beta = min(beta, 1)
        replay = self.replay_buffer.sample(self.batch_size, beta=beta)

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
        var_targets = var_targets.unsqueeze(-1).detach()

        # gradient descend model
        var_states0 = Variable(states0.float())
        var_action_values = self._model.forward(var_states0).gather(1, Variable(actions.view(-1, 1)))
        # var_loss = self._loss(var_action_values, var_targets)
        var_loss = torch.abs(var_action_values - var_targets)

        if self.t % 2000 == 0:
            print(var_targets)
        #     pdb.set_trace()

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
        loss = 0
        episode_rewards = [0]
        while self.t < self.max_timesteps:
            # play
            for i in range(4):
                action = self.make_action(state0, False)
                state1, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(state0, action,
                                       float(reward), state1, float(done))
                # accumulate episode reward
                episode_rewards[-1] += reward

                # update previous state
                if done:
                    state0 = self.env.reset()
                    print('t = %d, r = %f, loss = %f, exp = %f'
                          % (self.t, episode_rewards[-1], loss, self.epsilon))
                    episode_rewards.append(0)
                else:
                    state0 = state1

            # if self.t > 1000:
            #     # train on batch
            loss = self.update_model(target_q)

            # update target network
            if self.t % self.target_network_update_freq == 0:
                target_q.load_state_dict(self._model.state_dict())

            if self.t % 5000 == 0:
                torch.save({
                    'model': self._model.state_dict()
                }, 'model')

            self.t += 1
