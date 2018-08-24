# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Code adapted from https://github.com/phillipi/pix2pix/blob/master/datasets/download_dataset.sh
FILE=edges2shoes # edges2handbags
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./$FILE.tar.gz
TARGET_DIR=./$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./
rm $TAR_FILE

python process_data.py $FILE
rm -r $TARGET_DIR
