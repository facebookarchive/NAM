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
import collections
import torch
import torch.nn as nn
import conditional
import unconditional
import perceptual_loss


NAMParams = collections.namedtuple('NAMParams', 'nz ngf mu sd force_l2')
NAMParams.__new__.__defaults__ = (None, None, None, None, None)
OptParams = collections.namedtuple('OptParams', 'lr lr_ratio batch_size ' +
                                                'epochs ' +
                                                'decay_epochs decay_rate')
OptParams.__new__.__defaults__ = (None, None, None, None, None, None)


def distance_metric(sz, nc, force_l2=False):
    if nc == 1 or force_l2:
        return nn.MSELoss().cuda()
    elif sz == 32:
        return perceptual_loss._VGGDistance(3)
    elif sz == 64:
        return perceptual_loss._VGGDistance(4)
    else:
        assert False, "Perceptual loss only supports 32X32 and 64X64 images."


def transformer(sz, ngf, nc_in, nc_out):
    assert (sz == 32 or sz == 64), "Input must be 32X32 or 64X64"
    if sz == 32:
        net = conditional._netT32(ngf, nc_in, nc_out)
    elif sz == 64:
        net = conditional._netT64(ngf, nc_in, nc_out)
    net.apply(conditional.weights_init)
    return net


def generator(sz, nz, ngf, nc):
    assert (sz == 32 or sz == 64), "Input must be 32X32 or 64X64"
    if sz == 32:
        net = unconditional._netG32(nz, ngf, nc)
    elif sz == 64:
        net = unconditional._netG64(nz, ngf, nc)
    net.apply(unconditional.weights_init)
    return net


def discriminator(sz, nz, ndf, nc):
    assert (sz == 32 or sz == 64), "Input must be 32X32 or 64X64"
    if sz == 32:
        net = unconditional._netD32(nz, ndf, nc)
    elif sz == 64:
        net = unconditional._netD64(nz, ndf, nc)
    net.apply(unconditional.weights_init)
    return net


def latent_codes(nz, n):
    net = conditional._netZ(nz, n)
    net.apply(conditional.weights_init)
    return net


def unnorm(ims, mu, sd):
    for i in range(min(len(mu), ims.shape[1])):
        ims[:, i] = ims[:, i] * sd[i]
        ims[:, i] = ims[:, i] + mu[i]
    return ims


def format_im(ims_gen, mu, sd):
    ims_gen = ims_gen.clone()
    ims_gen = unnorm(ims_gen, mu, sd)
    ims_gen = torch.clamp(ims_gen.data, 0, 1)
    return ims_gen
