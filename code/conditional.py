# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Emb') != -1:
        # init.xavier_uniform(m.weight, gain=np.sqrt(2.0))
        init.normal(m.weight, mean=0, std=0.01)


class _netZ(nn.Module):
    def __init__(self, nz, n):
        super(_netZ, self).__init__()
        self.n = n
        self.emb = nn.Embedding(self.n, nz)
        self.nz = nz

    def get_norm(self):
        wn = self.emb.weight.norm(2, 1).data.unsqueeze(1)
        self.emb.weight.data = \
            self.emb.weight.data.div(wn.expand_as(self.emb.weight.data))

    def forward(self, idx):
        z = self.emb(idx).squeeze()
        return z


class _netT64(nn.Module):
    def __init__(self, ngf, in_nc, out_nc):
        super(_netT64, self).__init__()

        def normlayer(x):
            # return nn.InstanceNorm2d(x)
            return nn.BatchNorm2d(x)
            # return nn.Identity()
        # 4 X 4
        self.conv11 = nn.Conv2d(in_nc, 4 * ngf, 3, 1, 1, bias=False)
        self.bn11 = normlayer(4 * ngf)
        # 8 X 8
        self.conv21 = nn.Conv2d(4 * ngf + in_nc, 4 * ngf, 3, 1, 1, bias=False)
        self.bn21 = normlayer(4 * ngf)
        # 16 X 16
        self.conv31 = nn.Conv2d(4 * ngf + in_nc, 4 * ngf, 3, 1, 1, bias=False)
        self.bn31 = normlayer(4 * ngf)
        # 32 X 32
        self.conv41 = nn.Conv2d(4 * ngf + in_nc, 2 * ngf, 3, 1, 1, bias=False)
        self.bn41 = normlayer(2 * ngf)
        # 64 X 64
        self.conv51 = nn.Conv2d(2 * ngf + in_nc, ngf, 3, 1, 1, bias=False)
        self.bn51 = normlayer(ngf)
        self.conv52 = nn.Conv2d(ngf, out_nc, 3, 1, 1, bias=False)

        self.us = nn.Upsample(scale_factor=2, mode='bilinear')
        self.tanh = nn.Tanh()
        # self.nonlin = nn.SELU(True)
        self.nonlin = nn.LeakyReLU(0.2, True)
        self.ss = nn.AvgPool2d(2, stride=2)

    def main(self, I64):
        I32 = self.ss(I64)
        I16 = self.ss(I32)
        I8 = self.ss(I16)
        I4 = self.ss(I8)
        z = self.conv11(I4)
        z = self.bn11(z)
        z = self.nonlin(z)
        z = self.us(z)
        z = torch.cat((z, I8), 1)
        z = self.conv21(z)
        z = self.bn21(z)
        z = self.nonlin(z)
        z = self.us(z)
        z = torch.cat((z, I16), 1)
        z = self.conv31(z)
        z = self.bn31(z)
        z = self.nonlin(z)
        z = self.us(z)
        z = torch.cat((z, I32), 1)
        z = self.conv41(z)
        z = self.bn41(z)
        z = self.nonlin(z)
        z = self.us(z)
        z = torch.cat((z, I64), 1)
        z = self.conv51(z)
        z = self.bn51(z)
        z = self.nonlin(z)
        z = self.conv52(z)
        # z = nn.Sigmoid()(z)
        return z

    def forward(self, Is):
        HR = self.main(Is)
        return HR


class _netT32(nn.Module):
    def __init__(self, ngf, in_nc, out_nc):
        super(_netT32, self).__init__()

        def normlayer(x):
            # return nn.InstanceNorm2d(x)
            return nn.BatchNorm2d(x)
            # return nn.Identity()
        # 4 X 4
        self.conv11 = nn.Conv2d(in_nc, 4 * ngf, 3, 1, 1, bias=False)
        self.bn11 = normlayer(4 * ngf)
        # 8 X 8
        self.conv21 = nn.Conv2d(4 * ngf + in_nc, 3 * ngf, 3, 1, 1, bias=False)
        self.bn21 = normlayer(3 * ngf)
        # 16 X 16
        self.conv31 = nn.Conv2d(3 * ngf + in_nc, 2 * ngf, 3, 1, 1, bias=False)
        self.bn31 = normlayer(2 * ngf)
        # 64 X 64
        self.conv41 = nn.Conv2d(2 * ngf + in_nc, ngf, 3, 1, 1, bias=False)
        self.bn41 = normlayer(ngf)
        self.conv42 = nn.Conv2d(ngf, out_nc, 3, 1, 1, bias=False)

        self.us = nn.Upsample(scale_factor=2, mode='bilinear')
        self.tanh = nn.Tanh()
        # self.nonlin = nn.SELU(True)
        self.nonlin = nn.LeakyReLU(0.2, True)
        self.ss = nn.AvgPool2d(2, stride=2)

    def main(self, I32):
        I16 = self.ss(I32)
        I8 = self.ss(I16)
        I4 = self.ss(I8)
        z = self.conv11(I4)
        z = self.bn11(z)
        z = self.nonlin(z)
        z = self.us(z)
        z = torch.cat((z, I8), 1)
        z = self.conv21(z)
        z = self.bn21(z)
        z = self.nonlin(z)
        z = self.us(z)
        z = torch.cat((z, I16), 1)
        z = self.conv31(z)
        z = self.bn31(z)
        z = self.nonlin(z)
        z = self.us(z)
        z = torch.cat((z, I32), 1)
        z = self.conv41(z)
        z = self.bn41(z)
        z = self.nonlin(z)
        z = self.conv42(z)
        z = self.tanh(z)
        return z

    def forward(self, Is):
        HR = self.main(Is)
        return HR
