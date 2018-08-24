# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import numpy as np
import torch
import opt_nam
import utils
import params


# Load pretrained unconditional generator for X domain.
# Change code if you wish to use your own generative model.
def get_x_generator():
    netG = utils.generator(params.uncon_shape[0],
                           params.nz,
                           params.uncon_ngf,
                           params.uncon_shape[2])
    state_dict = torch.load(params.uncon_cp_path)
    netG.load_state_dict(state_dict)
    return netG


netG = get_x_generator()
netG.cuda()


# load Y domain data. Assumed to be a uint8 numpy array
# of shape (n, sz, sz, nc)
def get_y_data():
    data_np = np.load(params.y_data_path)
    data_np = data_np.transpose((0, 3, 1, 2))
    data_np = data_np / 255.0
    return data_np


y = get_y_data()
for i in range(y.shape[1]):
    y[:, i:i+1] -= params.mu[i]
    y[:, i:i+1] /= params.sd[i]
rp = np.random.permutation(y.shape[0])[:params.n_examples]
image_id = int(sys.argv[1]) % y.shape[0]
y = y[image_id]
y = np.tile(y[None, :], (params.batch_size, 1, 1, 1))


def get_transformer(netT_pth):
    netT = utils.transformer(y.shape[2], params.nam_ngf,
                             params.uncon_shape[2], y.shape[1])
    state_dict = torch.load(netT_pth)
    netT.load_state_dict(state_dict)
    return netT


netT = get_transformer(params.transformer_cp_path)
netT.cuda()

net_params = utils.NAMParams(nz=params.nz,
                             ngf=params.nam_ngf,
                             mu=params.mu,
                             sd=params.sd,
                             force_l2=False)
opt_params = utils.OptParams(lr=params.lr,
                             batch_size=params.batch_size,
                             epochs=params.epochs,
                             decay_epochs=params.decay_epochs,
                             decay_rate=params.decay_rate)

nm = opt_nam.NAMTrainer(netG, params.uncon_shape, net_params)
nm.eval_nam(netT, y, opt_params)
