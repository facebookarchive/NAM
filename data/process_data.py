# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import numpy as np
import scipy.misc as misc

dataset_name = sys.argv[1]
fn = "%s/train" % dataset_name
ls = os.listdir(fn)
n = len(ls)
x = np.zeros((n, 64, 64, 3), dtype="uint8")
y = np.zeros((n, 64, 64, 3), dtype="uint8")

for i in range(n):
    im = misc.imread("%s/train/%s" % (dataset_name, ls[i]))
    im = misc.imresize(im, (64, 128))
    x[i] = im[:, 64:]
    y[i] = im[:, :64]

np.save("%s_x" % dataset_name, x)
np.save("%s_y" % dataset_name, y)
