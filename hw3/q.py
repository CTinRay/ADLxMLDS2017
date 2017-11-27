import torch


class Q(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Q, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[-1], 32, 8, stride=4),
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
            torch.nn.Linear(3136, n_actions))

    def forward(self, frames):
        frames = frames.transpose(-3, -1)
        x = self.cnn(frames)
        x = x.view(x.size(0), -1)
        return self.mlp(x)
