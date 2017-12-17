import pdb
import torch
from torch.autograd import Variable
from nets import GeneratorNet, DiscriminatorNet


class GAN:
    def __init__(self,
                 dim_condition,
                 dim_noise=100,
                 learning_rate=0.0002,
                 batch_size=64,
                 max_epochs=300,
                 n_iters_d=4,
                 use_cuda=None):
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._n_iters_d = n_iters_d
        self._use_cuda = use_cuda
        if self._use_cuda is None:
            self._use_cuda = torch.cuda.is_available()

        self._generator = GeneratorNet(dim_condition, dim_noise)
        self._discriminator = DiscriminatorNet(dim_condition)

        if self._use_cuda:
            self._generator = self._generator.cuda()
            self._discriminator = self._discriminator.cuda()

        self._optimizer_g = torch.optim.RMSprop(
            self._generator.parameters(),
            lr=0.0002
        )
        self._optimizer_d = torch.optim.RMSprop(
            self._discriminator.parameters(),
            lr=0.0002)

    def train(self, real_dataset, fake_dataset):
        data_real = torch.utils.data.DataLoader(
            real_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=1)
        data_fake = torch.utils.data.DataLoader(
            fake_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=1)

        for epoch in range(self._max_epochs):
            for batch_real in data_real:
                batch_fake = iter(data_fake).next()
                for iter_d in range(self._n_iters_d):
                    condition_real = Variable(batch_real['label'])
                    img_real = Variable(batch_real['img'])
                    condition_fake = Variable(batch_fake['label'])
                    img_fake = Variable(batch_fake['img'])

                    if self._use_cuda:
                        condition_real = condition_real.cuda()
                        img_real = img_real.cuda()
                        condition_fake = condition_fake.cuda()
                        img_fake = img_fake.cuda()

                    img_gen = self._generator.forward(condition_real)

                    d_real = self._discriminator(img_real, condition_real)
                    d_gen = self._discriminator(img_gen, condition_real)
                    d_fake = self._discriminator(img_fake, condition_fake)

                    loss_d = 0.5 * torch.mean(d_fake) \
                        + 0.5 * torch.mean(d_gen) \
                        - torch.mean(d_real)

                    self._optimizer_d.zero_grad()
                    loss_d.backward()
                    self._optimizer_d.step()

                img_gen = self._generator.forward(condition_real)
                d_gen = self._discriminator(img_gen, condition_real)

                loss_g = torch.mean(d_gen)

                self._optimizer_g.zero_grad()
                loss_g.backward()
                self._optimizer_g.step()
