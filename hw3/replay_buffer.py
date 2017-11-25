import pdb
import numpy as np


class ReplayBuffer:
    def __init__(self, size, alpha=1):
        self.states0 = np.array([None] * size)
        self.states1 = np.array([None] * size)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.actions = np.zeros(size, dtype=int)
        self.dones = np.zeros(size, dtype=np.float32)
        self.priorities = np.zeros(size)
        self.max_size = size
        self.next_index = 0
        self.size = 0
        self._alpha = alpha
        self._max_priority = 1.0

    def add(self, state0, action, reward, state1, done):
        # save experience
        self.states0[self.next_index] = state0
        self.actions[self.next_index] = action
        self.rewards[self.next_index] = reward
        self.states1[self.next_index] = state1
        self.dones[self.next_index] = done
        self.priorities[self.next_index] = self._max_priority ** self._alpha

        # update data structure parameters
        self.next_index = (self.next_index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n_samples, beta=1):
        # calculate sample probability
        sample_probs = self.priorities / np.sum(self.priorities)

        # sample indices according with sample probability
        indices = np.where(
            np.random.multinomial(
                1,
                sample_probs,
                n_samples) == 1)[1]

        # calculate weights
        weights = (sample_probs[indices] * self.size) ** (-beta)

        # normalize with max_weight
        min_prob = np.min(sample_probs[:self.size])
        max_weight = (min_prob * self.size) ** (-beta)
        weights /= max_weight

        # convert lazy frame into np array
        states0 = np.array(list(map(np.array, self.states0[indices])))
        states1 = np.array(list(map(np.array, self.states1[indices])))

        return \
            states0, \
            self.actions[indices], \
            self.rewards[indices], \
            states1, \
            self.dones[indices], \
            indices, \
            weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities ** self._alpha
        self._max_priority = max(self._max_priority, np.max(priorities))
