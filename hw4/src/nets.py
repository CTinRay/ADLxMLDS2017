import pdb
import torch
from torch.autograd import Variable


class GeneratorNet(torch.nn.Module):
    def __init__(self, dim_condition, dim_noise):
        super(GeneratorNet, self).__init__()
        self.register_buffer('noise_mean',
                             torch.zeros(dim_noise))
        self.register_buffer('noise_std',
                             torch.ones(dim_noise))
        self.deconv_layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(dim_condition + dim_noise,
                                     512, 4, 1, 0),
            # 4 x 4
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1),
            # 8 x 8
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            # 16 x 16
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            # 32 x 32
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1),
            # 64 x 64
            torch.nn.Tanh()
        )

    def forward(self, condition):
        batch_size = condition.size(0)
        noise = torch.normal(
            torch.stack([Variable(self.noise_mean)] * batch_size, dim=0),
            torch.stack([Variable(self.noise_std)] * batch_size, dim=0)
        )
        net_input = torch.cat([condition, noise], dim=1)
        # [batch, dim_condition + dim_noise]

        net_input = net_input.unsqueeze(-1).unsqueeze(-1)
        # [batch, dim_condition + dim_noise, 1, 1]

        net_output = self.deconv_layers(net_input)
        return net_output


class DiscriminatorNet(torch.nn.Module):
    def __init__(self, dim_condition):
        super(DiscriminatorNet, self).__init__()
        self._dim_condition = dim_condition
        self.conv_layers1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 4, 2, 1),
            # 32 x 32
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, 4, 2, 1),
            # 16 x 16
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, 4, 2, 1),
            # 8 x 8
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 256, 4, 2, 1),
            # 4 x 4
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU()
        )

        self.conv_layers2 = torch.nn.Sequential(
            torch.nn.Conv2d(256 + dim_condition, 128, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 1, 4)
        )

    def forward(self, img, condition):
        batch_size = img.shape[0]
        condition = torch.stack([condition] * 16, dim=-1)

        conv_output = self.conv_layers1(img).view(batch_size, -1, 16)
        conv_output = torch.cat([conv_output, condition], dim=1)
        conv_output = conv_output.view(batch_size, -1, 4, 4)
        conv_output = self.conv_layers2(conv_output)
        conv_output = conv_output.view(batch_size, 1)
        return conv_output
