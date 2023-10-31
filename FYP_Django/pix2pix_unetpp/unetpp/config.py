# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:54
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : config.py
"""

"""
import os


class UNetConfig:

    def __init__(self,
                 epochs = 100,  # Number of epochs
                 batch_size = 32,    # Batch size
                 validation = 10.0,   # Percent of the data that is used as validation (0-100)
                 out_threshold=0.00325,

                 optimizer='Adam',
                 lr = 0.0001,     # learning rate
                 lr_decay_milestones = [20, 50],
                 lr_decay_gamma = 0.9,
                 weight_decay=1e-8,
                 momentum=0.9,
                 nesterov=True,

                 backbone = "efficientnet-b1",
                 n_channels = 3, # Number of channels in input images
                 n_classes = 9,  # Number of classes in the segmentation
                 scale = 1,    # Downscaling factor of the images
                 height = 256,  # Resize size of images
                 width = 256,

                 load = False,   # Load model from a .pth file
                 save_cp = True,

                 model='NestedUNet',
                 bilinear = True,
                 deepsupervision = True,
                 ):
        super(UNetConfig, self).__init__()

        # self.images_dir = './data/images'
        # self.masks_dir = './data/masks'
        self.checkpoints_dir = './pix2pix_unetpp/unetpp/data/checkpoints'

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold

        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.backbone = backbone
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale = scale
        self.height = height
        self.width = width

        self.load = load
        self.save_cp = save_cp

        self.model = model
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        os.makedirs(self.checkpoints_dir, exist_ok=True)
