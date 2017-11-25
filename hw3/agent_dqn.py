import numpy as np
import torch
from replay_buffer import ReplayBuffer
from pytorch_q import TorchQ


class AgentDQN:
    def __init__(self, env, args):
        self.env = env
        self.model = TorchQ(self.env.observation_space.shape,
                            self.env.action_space.n)
        self.n_actions = self.env.action_space.n
        self.max_timesteps = args['max_timesteps']
        self.gamma = args['gamma']
        self.replay_buffer = ReplayBuffer(args['buffer_size'])
        self.t = 0
        self.exploration_final_eps = args['exploration_final_eps']
        self.batch_size = args['batch_size']
        self.prioritized_replay_eps = args['prioritized_replay_eps']
        self.target_network_update_freq = args['target_network_update_freq']

    def init_game_setting(self):
        pass

    def make_action(self, state, test):
        state = np.expand_dims(state, 0)
        state = torch.from_numpy(state).float()
        action_value = self.model._predict_batch(state).data.cpu().numpy()
        best_action = np.argmax(action_value, -1).reshape((1,))

        # decide if doing exploration
        if not test:
            epsilon = 1 \
                - (1 - self.exploration_final_eps) \
                * self.t / self.max_timesteps
            explore = np.random.binomial(1, epsilon, 1)
        else:
            explore = 0

        if explore:
            return np.random.randint(0, self.n_actions, 1)
        else:
            return best_action

    def update_model(self, target_q):
        # sample from replay_buffer
        beta = 1
        states0, actions, rewards, states1, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size, beta)

        states0 = torch.from_numpy(states0).float()
        states1 = torch.from_numpy(states1).float()
        actions = torch.from_numpy(actions).long()
        weights = torch.from_numpy(weights).float()

        # predict target with target network
        targets = rewards
        target_reward = \
            target_q._predict_batch(states1).data.max(-1)[1].cpu().numpy()
        targets += self.gamma * target_reward * (~dones)
        targets = torch.from_numpy(targets).float()

        # gradient descend model
        _, loss = self.model._run_iter(
            (states0, actions, targets, weights), True)
        loss = loss.data.cpu().numpy()

        # update experience priorities
        new_priority = loss + self.prioritized_replay_eps
        self.replay_buffer.update_priorities(indices, new_priority)

        return np.mean(loss)

    def train(self):
        target_q = TorchQ(self.env.observation_space.shape,
                          self.env.action_space.n)
        target_q._model.load_state_dict(self.model._model.state_dict())

        state0 = self.env.reset()

        # log statics
        episode_rewards = [0]
        for self.t in range(self.max_timesteps):
            # play
            action = self.make_action(np.array(state0), False)
            state1, reward, done, _ = self.env.step(action)
            self.replay_buffer.add(state0, action, reward, state1, done)
            # accumulate episode reward
            episode_rewards[-1] += reward

            # update previous state
            if done:
                state0 = self.env.reset()
                episode_rewards.append(0)
                print(episode_rewards[-2])
            else:
                state0 = state1

            # train on batch
            loss = self.update_model(target_q)

            # update target network
            if self.t % self.target_network_update_freq == 0:
                target_q._model.load_state_dict(self.model._model.state_dict())
