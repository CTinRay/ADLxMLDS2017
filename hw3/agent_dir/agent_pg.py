import torch
from torch.autograd import Variable


class Agent_PG():
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """        
        self.env = env
        self.batch_size = args.batch_size
        self._max_iters = args.max_timesteps
        self._iter = 0

        self._model = Policy(self.env.observation_space.shape,
                             self.env.action_space.n)
        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=args.learning_rate)
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            self._model = self._model.cuda()

        if args.test_pg:
            print('loading trained model')
            if self._use_cuda:
                ckp = torch.load(args.test_pg)
            else:
                ckp = torch.load(args.test_pg,
                                 map_location=lambda storage, loc: storage)

            self._model.load_state_dict(ckp['model'])


        self._prev_obs = None

    def init_game_setting(self):
        pass

    def _train_iteration(self, mean):
        step = 0
        total_log_probs = 0
        self._prev_obs = self.env.reset()
        obs, reward, done, _ = self.env.step(0)
        total_rewards = reward
        while not done:
            # calculate action probability
            var_obs = Variable(torch.from_numpy(obs - self._prev_obs)
                               .float().unsqueeze(0))
            var_obs = var_obs.max(-1)[0]
            if self._use_cuda:
                var_obs = var_obs.cuda()
            action_probs = self._model.forward(var_obs)

            # sample action
            action = torch.multinomial(action_probs, 1).data

            # get next observation
            self._prev_obs = obs
            obs, reward, done, _ = self.env.step(action[0, 0])

            # accumulate reward and probability
            total_rewards += reward
            total_log_probs += torch.log(action_probs[:, action])
            step += 1

        self._optimizer.zero_grad()
        (-(total_rewards - mean) * total_log_probs).backward()
        self._optimizer.step()

        return obs, total_rewards

    def train(self):
        rewards = [-21]
        while self._iter < self._max_iters:
            mean = sum(rewards[-20:]) / len(rewards[-20:])
            obs, reward = self._train_iteration(mean)
            rewards.append(reward)

            if self._iter % 20 == 0:
                print('%d %f' % (self._iter, mean))

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
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()


class Policy(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Policy, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 8, stride=4),
            torch.nn.ELU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ELU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ELU(),
            # torch.nn.MaxPool2d(2)
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(22528, n_actions),
            torch.nn.Softmax()
        )

    def forward(self, frames):
        frames = frames.unsqueeze(-3)
        x = self.cnn(frames)
        x = x.view(x.size(0), -1)
        return self.mlp(x)
