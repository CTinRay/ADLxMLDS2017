import torch


class Q(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Q, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[-1], 16, 3, padding=1),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(2))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                32 * input_shape[0] // 4 * input_shape[1] // 4,
                256),
            torch.nn.Linear(256, n_actions))

    def forward(self, frames):
        x = self.cnn(frames)
        x = x.view(x.size(0), -1)
        return self.mlp(x)
