# Adapted from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch.nn as nn

# Generator Code
# class Generator(nn.Module):
#     def __init__(self, ngpu, ngf, nz, image_dim, nc):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         if image_dim == 64:
#           self.main = nn.Sequential(
#               # input is Z, going into a convolution
#               nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
#               nn.BatchNorm2d(ngf * 8),
#               nn.ReLU(True),
#               # state size. (ngf*8) x 4 x 4
#               nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ngf * 4),
#               nn.ReLU(True),
#               # state size. (ngf*4) x 8 x 8
#               nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ngf * 2),
#               nn.ReLU(True),
#               # state size. (ngf*2) x 16 x 16
#               nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ngf),
#               nn.ReLU(True),
#               # state size. (ngf) x 32 x 32
#               nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
#               #nn.Tanh()
#               nn.Sigmoid()
#               # state size. (nc) x 64 x 64
#           )
#         elif image_dim == 128:
#           self.main = nn.Sequential(
#               # input is Z, going into a convolution
#               nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
#               nn.BatchNorm2d(ngf * 8),
#               nn.ReLU(True),
#               # state size. (ngf*8) x 4 x 4
#               nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ngf * 4),
#               nn.ReLU(True),
#               # state size. (ngf*4) x 8 x 8
#               nn.ConvTranspose2d( ngf * 4, ngf * 4, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ngf * 4),
#               nn.ReLU(True),
#               # state size. (ngf*2) x 16 x 16
#               nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ngf * 2),
#               nn.ReLU(True),
#               # state size. (ngf) x 32 x 32
#               nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ngf),
#               nn.ReLU(True),
#               # state size. (ngf) x 32 x 32
#               nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
#               #nn.Tanh()
#               nn.Sigmoid()
#               # state size. (nc) x 64 x 64
#           )
#         else:
#           raise NotImplementedError
#
#
#     def forward(self, input):
#         return self.main(input)
#
# class Discriminator(nn.Module):
#     def __init__(self, ngpu, ndf, nc, image_dim):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         if image_dim == 64:
#           self.main = nn.Sequential(
#               # input is (nc) x 64 x 64
#               nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#               nn.LeakyReLU(0.2, inplace=True),
#               # state size. (ndf) x 32 x 32
#               nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ndf * 2),
#               nn.LeakyReLU(0.2, inplace=True),
#               # state size. (ndf*2) x 16 x 16
#               nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ndf * 4),
#               nn.LeakyReLU(0.2, inplace=True),
#               # state size. (ndf*4) x 8 x 8
#               nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#               nn.BatchNorm2d(ndf * 8),
#               nn.LeakyReLU(0.2, inplace=True),
#               # state size. (ndf*8) x 4 x 4
#               nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#               nn.Sigmoid()
#           )
#         elif image_dim == 128:
#           self.main = nn.Sequential(
#                 # input is (nc) x 64 x 64
#                 nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # state size. (ndf) x 32 x 32
#                 nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#                 nn.BatchNorm2d(ndf * 2),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # state size. (ndf*2) x 16 x 16
#                 nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#                 nn.BatchNorm2d(ndf * 4),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # state size. (ndf*4) x 8 x 8
#                 nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#                 nn.BatchNorm2d(ndf * 8),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # state size. (ndf*8) x 4 x 4
#                 nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
#                 nn.BatchNorm2d(ndf * 8),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # state size. (ndf*8) x 4 x 4
#                 nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#                 nn.Sigmoid()
#             )
#         else:
#           raise NotImplementedError
#
#     def forward(self, input):
#         return self.main(input)

import numpy as np

class Generator(nn.Module):
    def __init__(self, ngpu, ngf, nz, image_dim, nc):
        super(Generator, self).__init__()

        self.image_dim = (nc, image_dim, image_dim)
        self.nz = nz

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.nz, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_dim))),
            # nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.image_dim)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_dim)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity