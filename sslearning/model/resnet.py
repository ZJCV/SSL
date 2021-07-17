# -*- coding: utf-8 -*-

"""
@date: 2021/6/11 上午11:25
@file: resnet.py
@author: zj
@description: 
"""

import torch.nn as nn
import torchvision.models as models

from zcls.config.key_word import KEY_OUTPUT


class ResNet(nn.Module):

    def __init__(self, num_classes=1000, arch='resnet50'):
        super().__init__()

        assert arch in ['resnet50', 'resnet101', 'resnet152']
        self.model = eval(f'models.{arch}')(pretrained=True)

        self.init_weight(num_classes)

    def init_weight(self, num_classes):
        if num_classes != 1000:
            old_fc = self.model.fc
            assert isinstance(old_fc, nn.Linear)

            in_features = old_fc.in_features
            new_fc = nn.Linear(in_features, num_classes, bias=True)
            nn.init.normal_(new_fc.weight, 0, 0.01)
            nn.init.zeros_(new_fc.bias)

            self.model.fc = new_fc

    def forward(self, x):
        res = self.model(x)

        return {KEY_OUTPUT: res}


def get_resnet(num_classes=1000, arch='resnet50'):
    return ResNet(num_classes=num_classes, arch=arch)
