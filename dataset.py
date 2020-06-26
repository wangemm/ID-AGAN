import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob


def normalize(data):
    return np.float32(data / 255.)


def img_to_patches(img, win, stride):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    res = np.zeros([endc, win * win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            res[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k = k + 1
    return res.reshape([endc, win, win, total_pat_num])


def prepare_data(data_path, val_data_path, patch_size=256, stride=20, gray_mode=True):
    print('> Training database')
    scales = [1, 0.8]
    types = ('*.bmp', '*.png', '*.jpg')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(data_path, tp)))
    files.sort()
    print(files)

    if gray_mode:
        traindbf = 'train_gray.h5'
        valdbf = 'val_gray.h5'
    else:
        traindbf = 'train_rgb.h5'
        valdbf = 'val_rgb.h5'

    train_num = 0
    i = 0
    with h5py.File(traindbf, 'w') as h5f:
        while i < len(files):
            imgor = cv2.imread(files[i])
            # h, w, c = img.shape
            for sca in scales:
                img = cv2.resize(imgor, (0, 0), fx=sca, fy=sca, \
                                 interpolation=cv2.INTER_CUBIC)
                if not gray_mode:
                    # CxHxW RGB image
                    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
                else:
                    # CxHxW grayscale image (C=1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img, 0)
                img = normalize(img)
                patches = img_to_patches(img, win=patch_size, stride=stride)
                print("\tfile: %s scale %.1f # samples: %d" % \
                      (files[i], sca, patches.shape[3]))
                for nx in range(patches.shape[3]):
                    data = patches[:, :, :, nx].copy()
                    h5f.create_dataset(str(train_num), data=data)
                    train_num += 1
            i += 1

    # validation database
    print('\n> Validation database')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(val_data_path, tp)))
    files.sort()
    h5f = h5py.File(valdbf, 'w')
    val_num = 0
    for i, item in enumerate(files):
        print("\tfile: %s" % item)
        img = cv2.imread(item)
        if not gray_mode:
            # C. H. W, RGB image
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            # C, H, W grayscale image (C=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 0)
        # img = normalize(img)
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()

    print('\n> Total')
    print('\ttraining set, # samples %d' % train_num)
    print('\tvalidation set, # samples %d\n' % val_num)


class Dataset(torch.utils.data.Dataset):
    r"""Implements torch.utils.data.Dataset
    """

    def __init__(self, train=True, gray_mode=True):
        super(Dataset, self).__init__()
        self.train = train
        self.gray_mode = gray_mode
        if not self.gray_mode:
            self.traindbf = 'train_rgb.h5'
            self.valdbf = 'val_rgb.h5'
        else:
            self.traindbf = 'train_gray.h5'
            self.valdbf = 'val_gray.h5'

        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
        else:
            h5f = h5py.File(self.valdbf, 'r')
        self.keys = list(h5f.keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
        else:
            h5f = h5py.File(self.valdbf, 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
