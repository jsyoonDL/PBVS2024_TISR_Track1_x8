# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 22:25:24 2024

@author: user
"""

import kornia.augmentation as K
import torch.nn as nn
from torch import Tensor
import torch


#%% DataAugmentation
class  DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.transforms = nn.Sequential(     
            # K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.05),
            K.RandomPerspective(0.5, p=0.5),
            K.RandomAffine((-15., 20.), p=0.5),
            # K.RandomContrast((0.8,1.2), p=0.5),
            # K.RandomElasticTransform(p=0.5),
            # K.RandomGaussianNoise(mean=0,std=0.05, p=0.5),
            # K.RandomSharpness(sharpness=0.25, p=0.5),
            # K.RandomClahe()
            )
        # self.cutmix = K.RandomCutMixV2(p=0.5)
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:

        x_out = self.transforms(x)  # BxCxHxW
        # x_out = self.cutmix(x_out)
        return x_out
