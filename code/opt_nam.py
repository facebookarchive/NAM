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
import numpy as np
import os
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import utils


class NAM():
    def __init__(self, net_params, netG, input_shape):
        self.net_params = net_params
        self.input_shape = input_shape

        self.netG = netG
        self.netZ = None
        self.netT = None

        self.mu = net_params.mu
        self.sd = net_params.sd

    def fit_to_target_domain(self, ims_np, opt_params, vis_epochs=10):
        n, nc, sz, sz_y = ims_np.shape
        assert (sz == sz_y), "Input must be square!"
        self.netZ = utils.latent_codes(self.net_params.nz, n)
        self.netZ.cuda()

        self.netT = utils.transformer(sz, self.net_params.ngf,
                                      self.input_shape[2], nc)
        self.netT.cuda()

        self.dist = utils.distance_metric(sz, nc, self.net_params.force_l2)

        for epoch in range(opt_params.epochs):
            er = self.train_epoch(epoch, ims_np, opt_params)
            print("NAM Epoch: %d Error: %f" % (epoch, er))
            torch.save(self.netZ.state_dict(), 'nam_nets/netZ.pth')
            torch.save(self.netT.state_dict(), 'nam_nets/netT.pth')
            if epoch % vis_epochs == 0:
                self.visualize(epoch, ims_np, "nam_train_ims")

    def train_epoch(self, epoch, ims_np, opt_params):
        n, nc, sz, _ = ims_np.shape
        rp = np.random.permutation(n)
        # Compute batch size
        batch_size = opt_params.batch_size
        batch_n = n // batch_size
        # Initialize tensors
        idx = torch.LongTensor(batch_size)
        idx = Variable(idx.cuda())
        images = torch.FloatTensor(batch_size, nc, sz, sz)
        images = Variable(images.cuda())
        # Compute learning rate
        decay_steps = epoch // opt_params.decay_epochs
        lr = opt_params.lr * opt_params.decay_rate ** decay_steps
        # Initialize optimizers
        optimizerT = optim.Adam(self.netT.parameters(),
                                lr=opt_params.lr_ratio*lr,
                                betas=(0.5, 0.999))
        optimizerZ = optim.Adam(self.netZ.parameters(), lr=lr,
                                betas=(0.5, 0.999))
        # Start optimizing
        er = 0
        for i in range(batch_n):
            # Put numpy data into tensors
            np_idx = rp[i * batch_size: (i + 1) * batch_size]
            idx.data.copy_(torch.from_numpy(np_idx))
            np_data = ims_np[rp[i * batch_size: (i + 1) * batch_size]]
            images.data.resize_(np_data.shape).copy_(torch.from_numpy(np_data))
            # Forward pass
            optimizerT.zero_grad()
            optimizerZ.zero_grad()
            zi = self.netZ(idx)
            I_implicit = self.netG(zi.view(batch_size, -1, 1, 1))
            I_target_est = self.netT(I_implicit)
            rec_loss = self.dist(I_target_est, images)
            # Backward pass and optimization step
            rec_loss.backward()
            optimizerT.step()
            optimizerZ.step()
            er += rec_loss.data[0]
        er = er / batch_n
        return er

    def eval_target_images(self, netT, ims_np, opt_params, vis_epochs=10):
        n, nc, sz, sz_y = ims_np.shape
        assert (sz == sz_y), "Input must be square!"
        self.netZ = utils.latent_codes(self.net_params.nz, n)
        self.netZ.cuda()
        self.netT = netT
        self.dist = utils.distance_metric(sz, nc, self.net_params.force_l2)

        for epoch in range(opt_params.epochs):
            er = self.eval_epoch(epoch, ims_np, opt_params)
            print("NAM Eval Epoch: %d Error: %f" % (epoch, er))
            if epoch % vis_epochs == 0:
                self.visualize(epoch, ims_np, "nam_eval_ims")

    def eval_epoch(self, epoch, ims_np, opt_params):
        n, nc, sz, _ = ims_np.shape
        # Compute batch size
        batch_size = opt_params.batch_size
        batch_n = n // batch_size
        # Initialize tensors
        idx = torch.LongTensor(batch_size)
        idx = Variable(idx.cuda())
        images = torch.FloatTensor(batch_size, nc, sz, sz)
        images = Variable(images.cuda())
        # Compute learning rate
        decay_steps = epoch // opt_params.decay_epochs
        lr = opt_params.lr * opt_params.decay_rate ** decay_steps
        # Initialize optimizers
        optimizerZ = optim.Adam(self.netZ.parameters(), lr=lr,
                                betas=(0.5, 0.999))
        # Start optimizing
        er = 0
        for i in range(batch_n):
            # Put numpy data into tensors
            np_idx = i * batch_size + np.arange(batch_size)
            idx.data.copy_(torch.from_numpy(np_idx))
            np_data = ims_np[np_idx]
            images.data.resize_(np_data.shape).copy_(torch.from_numpy(np_data))
            # Forward pass
            optimizerZ.zero_grad()
            zi = self.netZ(idx)
            I_implicit = self.netG(zi.view(batch_size, -1, 1, 1))
            I_target_est = self.netT(I_implicit)
            rec_loss = self.dist(I_target_est, images)
            # Backward pass and optimization step
            rec_loss.backward()
            optimizerZ.step()
            er += rec_loss.data[0]
        er = er / batch_n
        return er

    def visualize(self, epoch, ims_np, ims_dir, n_vis=64):
        idx = torch.from_numpy(np.arange(n_vis)).cuda()
        zi = self.netZ(Variable(idx)).view(n_vis, -1, 1, 1)
        I_implicit = self.netG(zi)
        I_est = self.netT(I_implicit)
        I_target = Variable(torch.from_numpy(ims_np[:n_vis]).float()).cuda()

        ims = utils.format_im(I_implicit, self.mu, self.sd)
        vutils.save_image(ims,
                          '%s/implicit_epoch_%03d.png' % (ims_dir, epoch),
                          normalize=False)
        ims = utils.format_im(I_est, self.mu, self.sd)
        vutils.save_image(ims,
                          '%s/est_epoch_%03d.png' % (ims_dir, epoch),
                          normalize=False)
        ims = utils.format_im(I_target, self.mu, self.sd)
        vutils.save_image(ims, '%s/target.png' % ims_dir, normalize=False)


class NAMTrainer():
    def __init__(self, netG, input_shape, net_params):
        if not os.path.isdir("nam_train_ims"):
            os.mkdir("nam_train_ims")
        if not os.path.isdir("nam_eval_ims"):
            os.mkdir("nam_eval_ims")
        if not os.path.isdir("nam_nets"):
            os.mkdir("nam_nets")
        sz, sz_y, nc = input_shape
        assert (sz == sz_y), "Input must be square!"
        assert ((sz == 32) or (sz == 64)), "Input must be 32X32 or 64X64!"
        self.nam = NAM(net_params, netG, input_shape)

    def train_nam(self, y_ims, opt_params):
        self.nam.fit_to_target_domain(y_ims, opt_params)

    def eval_nam(self, netT, y_ims, opt_params):
        self.nam.eval_target_images(netT, y_ims, opt_params)
