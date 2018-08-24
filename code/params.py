# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Data paths
x_data_path = "../data/edges2shoes_x.npy"
y_data_path = "../data/edges2shoes_y.npy"
# Unconditional generative model for X domain
uncon_ngf = 64
uncon_ndf = 64
uncon_batch_size = 64
uncon_shape = (64, 64, 3)
uncon_niter = 26
uncon_lr = 0.0002
uncon_cp_path = "unconditional_nets/netG_25.pth"
# NAM params
nz = 100
mu = [0.5, 0.5, 0.5]
sd = [0.5, 0.5, 0.5]
nam_ngf = 32
n_examples = 2000
lr = 3e-2
lr_ratio = 0.03
batch_size = 128
epochs = 51
decay_epochs = 50
decay_rate = 0.5
transformer_cp_path = "nam_nets/netT.pth"
