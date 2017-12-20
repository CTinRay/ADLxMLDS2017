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

        self._n_epochs = 0

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
            mean_loss_d = 0
            mean_loss_g = 0
            for batch_real in tqdm(data_real):
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

                    # clip weights
                    for p in self._discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    mean_loss_d += loss_d.data / self._n_iters_d \
                        * self._batch_size / len(fake_dataset)

                img_gen = self._generator.forward(condition_real)
                d_gen = self._discriminator(img_gen, condition_real)

                loss_g = -torch.mean(d_gen)

                self._optimizer_g.zero_grad()
                loss_g.backward()
                self._optimizer_g.step()

                mean_loss_g += loss_g.data \
                    * self._batch_size / len(fake_dataset)

            print('mean generator loss = {}, mean discriminator loss = {}'
                  .format(mean_loss_g[0],
                          mean_loss_d[0]))
            if epoch % 10 == 0:
                torch.save(
                    {'epoch': epoch,
                     'generator': self._generator.state_dict(),
                     'discriminator': self._discriminator.state_dict()},
                    'model-epoch-%d' % epoch)
                imgs = img_gen.data.cpu().transpose(-1, -3) \
                                         .numpy()
                skimage.io.imsave('img-epoch-%d.jpg' % epoch,
                                  imgs[0])

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
            batch_result = self._generator.forward(batch_condition) \
                                          .data.transpose(-1, -3)
            results.append(batch_result)

        results = torch.cat(results, dim=0)
        return results.cpu().numpy()
