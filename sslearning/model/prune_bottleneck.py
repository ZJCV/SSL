# -*- coding: utf-8 -*-

"""
@date: 2021/7/17 下午2:08
@file: prune_bottleneck.py
@author: zj
@description: 
"""

from torch import Tensor, nn as nn
from torchvision.models.resnet import Bottleneck


class PrunedBottleneck(nn.Module):

    def __init__(self, m) -> None:
        super(PrunedBottleneck, self).__init__()
        assert isinstance(m, Bottleneck)

        self.downsample = m.downsample

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        return x
