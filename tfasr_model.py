# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN


def swish(x):
    return F.relu(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.dconv2_2 = DCN(64, 64, 3, 1, 1)
        self.dconv2_3 = DCN(64, 64, 3, 1, 1)
        self.dconv2_4 = DCN(64, 1, 3, 1, 1)
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=0)

    def forward(self, x):
        #########################original version########################
        x = self.conv1(x)

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        x = swish(self.dconv2_2(x))
        x = swish(self.dconv2_3(x))
        return self.conv4(self.dconv2_4(x))
