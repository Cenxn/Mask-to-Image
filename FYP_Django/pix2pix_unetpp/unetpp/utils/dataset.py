# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:55
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : dataset.py
"""

"""
import os
import os.path as osp
import logging

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, height=256, width=256):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        # self.scale      = scale
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.height = height
        self.width = width
        assert height % 32 == 0 and width % 32 == 0, "image's size to be multiple of 32's"

        self.img_names = os.listdir(imgs_dir)
        logging.info(f'Creating dataset with {len(self.img_names)} examples')

    def __len__(self):
        return len(self.img_names)

    @classmethod
    def preprocess(cls, pil_img, height=256, width=256):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))
        pil_img = pil_img.resize((height, width))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            # mask target image
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans.astype(float)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        msk_name = img_name.split('.')[0] + '.png'
        img_path = osp.join(self.imgs_dir, img_name)
        mask_path = osp.join(self.masks_dir, msk_name)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # assert img.size == mask.size, \
        #     f'Image and mask {img_name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.height, self.width)
        mask = self.preprocess(mask, self.height, self.width)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
