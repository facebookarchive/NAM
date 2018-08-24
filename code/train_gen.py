# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Code adapted from:
# https://github.com/pytorch/examples/tree/master/dcgan
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import utils
import params


cudnn.benchmark = True


if not os.path.isdir("unconditional_ims"):
    os.mkdir("unconditional_ims")
if not os.path.isdir("unconditional_nets"):
    os.mkdir("unconditional_nets")


# load Y domain data. Assumed to be a uint8 numpy array
# of shape (n, sz, sz, nc)
def get_x_data():
    data_np = np.load(params.x_data_path)
    data_np = data_np.transpose((0, 3, 1, 2))
    data_np = data_np / 255.0
    return data_np


x = get_x_data()
x = (x - 0.5) / 0.5
n = len(x)

nz = params.nz
ngf = params.uncon_ngf
ndf = params.uncon_ndf
assert (params.uncon_shape[0] ==
        params.uncon_shape[1]), "Images must be square!"
sz = params.uncon_shape[0]
nc = params.uncon_shape[2]
batchSize = params.uncon_batch_size
lr = params.uncon_lr
niter = params.uncon_niter


netG = utils.generator(sz, nz, ngf, nc)
netD = utils.discriminator(sz, nz, ndf, nc)

criterion = nn.BCELoss()

input = torch.FloatTensor(batchSize, nc, sz, sz)
noise = torch.FloatTensor(batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(batchSize)
real_label = 1
fake_label = 0

netD.cuda()
netG.cuda()
criterion.cuda()
input, label = input.cuda(), label.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


n_batch = n // batchSize
for epoch in range(niter):
    rp = np.random.permutation(n)
    for i in range(n_batch):
        idx_np = rp[i * batchSize: (i + 1) * batchSize]
        real = Variable(torch.from_numpy(x[idx_np]).cuda()).cuda()
        real = real.float()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        input.resize_as_(real.data).copy_(real.data)
        label.resize_(batchSize).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # fake labels are real for generator cost
        labelv = Variable(label.fill_(real_label))
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, niter, i, n_batch,
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real.data,
                              'unconditional_ims/real_samples.png',
                              normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                              'unconditional_ims/fake_epoch_%03d.png' % epoch,
                              normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), 'unconditional_nets/netG_%02d.pth' % epoch)
