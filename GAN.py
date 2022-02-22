# Adapted from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch.nn as nn

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, ngf, nz, image_size, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        if image_size == 64:
          self.main = nn.Sequential(
              # input is Z, going into a convolution
              nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
              nn.BatchNorm2d(ngf * 8),
              nn.ReLU(True),
              # state size. (ngf*8) x 4 x 4
              nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ngf * 4),
              nn.ReLU(True),
              # state size. (ngf*4) x 8 x 8
              nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ngf * 2),
              nn.ReLU(True),
              # state size. (ngf*2) x 16 x 16
              nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ngf),
              nn.ReLU(True),
              # state size. (ngf) x 32 x 32
              nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
              #nn.Tanh()
              nn.Sigmoid()
              # state size. (nc) x 64 x 64
          )
        elif image_size == 128:
          self.main = nn.Sequential(
              # input is Z, going into a convolution
              nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
              nn.BatchNorm2d(ngf * 8),
              nn.ReLU(True),
              # state size. (ngf*8) x 4 x 4
              nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ngf * 4),
              nn.ReLU(True),
              # state size. (ngf*4) x 8 x 8
              nn.ConvTranspose2d( ngf * 4, ngf * 4, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ngf * 4),
              nn.ReLU(True),
              # state size. (ngf*2) x 16 x 16
              nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ngf * 2),
              nn.ReLU(True),
              # state size. (ngf) x 32 x 32
              nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ngf),
              nn.ReLU(True),
              # state size. (ngf) x 32 x 32
              nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
              #nn.Tanh()
              nn.Sigmoid()
              # state size. (nc) x 64 x 64
          )
        else:
          raise NotImplementedError


    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc, image_size):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if image_size == 64:
          self.main = nn.Sequential(
              # input is (nc) x 64 x 64
              nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
              nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf) x 32 x 32
              nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ndf * 2),
              nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf*2) x 16 x 16
              nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ndf * 4),
              nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf*4) x 8 x 8
              nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ndf * 8),
              nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf*8) x 4 x 4
              nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
              nn.Sigmoid()
          )
        elif image_size == 128:
          self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        else:
          raise NotImplementedError

    def forward(self, input):
        return self.main(input)