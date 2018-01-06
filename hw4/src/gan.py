import numpy as np
import os
import pdb
import skimage.io
import torch
from tqdm import tqdm
from torch.autograd import Variable
from nets import GeneratorNet, DiscriminatorNet


class GAN:
    def __init__(self,
                 dim_condition,
                 dim_noise=100,
                 learning_rate=0.0002,
                 batch_size=64,
                 max_epochs=300,
                 g_update_interval=5,
                 lambda_grad=10,
                 save_interval=5,
                 save_dir='./',
                 cuda_rand_seed=None,
                 use_cuda=None):
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._g_update_interval = g_update_interval
        self._lambda_grad = lambda_grad
        self._use_cuda = use_cuda
        if self._use_cuda is None:
            self._use_cuda = torch.cuda.is_available()

        self._generator = GeneratorNet(dim_condition, dim_noise)
        self._discriminator = DiscriminatorNet(dim_condition)

        if self._use_cuda:
            self._generator = self._generator.cuda()
            self._discriminator = self._discriminator.cuda()
            if cuda_rand_seed is not None:
                torch.cuda.manual_seed_all(cuda_rand_seed)

        self._optimizer_g = torch.optim.RMSprop(
            self._generator.parameters(),
            lr=0.0002)
        self._optimizer_d = torch.optim.RMSprop(
            self._discriminator.parameters(),
            lr=0.0002)

        self._n_epochs = 0
        self._save_interval = save_interval
        self._save_dir = save_dir

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
            loss_ds = []
            loss_gs = []
            for i, batch_real, batch_fake in \
                tqdm(zip(range(len(data_real)),
                         data_real, data_fake)):

                # train discriminator
                condition_real = Variable(batch_real['label'])
                img_real = Variable(batch_real['img'])
                condition_fake = Variable(batch_fake['label'])
                img_fake = Variable(batch_fake['img'])
                alphas = Variable(torch.rand(img_real.shape[0], 1, 1, 1))
                ones = torch.ones(img_real.shape[0], 1)

                if self._use_cuda:
                    condition_real = condition_real.cuda()
                    img_real = img_real.cuda()
                    condition_fake = condition_fake.cuda()
                    img_fake = img_fake.cuda()
                    alphas = alphas.cuda()
                    ones = ones.cuda()

                img_gen = self._generator.forward(condition_real)

                d_real = self._discriminator(img_real, condition_real)
                d_gen = self._discriminator(img_gen, condition_real)
                d_fake = self._discriminator(img_fake, condition_fake)

                # interpolates = img_real * alphas + img_gen * (1 - alphas)
                # d_interpolates = self._discriminator(interpolates,
                #                                      condition_real)
                # gradients = torch.autograd.grad(d_interpolates,
                #                                 interpolates,
                #                                 grad_outputs=ones,
                #                                 create_graph=True)[0]
                # grad_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

                d_false = torch.mean(torch.cat([d_fake, d_gen], dim=1),
                                     dim=1)
                loss_d = torch.mean(d_false) \
                    - torch.mean(d_real) \
                    # + self._lambda_grad * grad_penalty

                for p in self._discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                self._optimizer_d.zero_grad()
                loss_d.backward()
                self._optimizer_d.step()

                loss_ds.append(loss_d.data[0])

                # train generator
                if i % self._g_update_interval == 0:
                    img_gen = self._generator.forward(condition_real)
                    d_gen = self._discriminator(img_gen, condition_real)

                    loss_g = -torch.mean(d_gen)

                    self._optimizer_g.zero_grad()
                    loss_g.backward()
                    self._optimizer_g.step()

                    loss_gs.append(loss_g.data[0])

            print('{} mean generator loss = {}, mean discriminator loss = {}'
                  .format(epoch,
                          sum(loss_gs) / len(loss_gs),
                          sum(loss_ds) / len(loss_ds)))

            if epoch % self._save_interval == 0:
                filename = os.path.join(self._save_dir,
                                        'model-epoch-%d' % epoch)
                torch.save(
                    {'epoch': epoch,
                     'generator': self._generator.state_dict(),
                     'discriminator': self._discriminator.state_dict()},
                    filename)
                imgs = ((img_gen.data.cpu().numpy() + 1) * 128).astype('uint8')
                print(condition_real[0])
                skimage.io.imsave('img-epoch-%d.jpg' % epoch,
                                  np.transpose(imgs[0], [1, 2, 0]))

    def load(self, filename):
        ckp = torch.load(filename)
        self._generator.load_state_dict(ckp['generator'])
        self._discriminator.load_state_dict(ckp['discriminator'])

    def inference(self, condition, batch_size=64):
        condition = torch.from_numpy(condition).float()
        results = []
        for b in range(0, condition.shape[0], batch_size):
            batch_condition = condition[b: b + batch_size]
            batch_condition = Variable(batch_condition)
            if self._use_cuda:
                batch_condition = batch_condition.cuda()
            batch_result = self._generator.forward(batch_condition).data
            results.append(batch_result)

        results = torch.cat(results, dim=0)
        return results.cpu().numpy()
