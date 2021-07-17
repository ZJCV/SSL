# -*- coding: utf-8 -*-

"""
@date: 2021/6/10 下午7:12
@file: vggnet.py
@author: zj
@description: 
"""

import torch.nn as nn
import torchvision.models as models

from zcls.config.key_word import KEY_OUTPUT


class VGGNet(nn.Module):

    def __init__(self, num_classes=1000, arch='vgg16_bn'):
        super().__init__()

        assert arch in ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
        self.model = eval(f'models.{arch}')(pretrained=True)

        self.init_weight(num_classes)

    def init_weight(self, num_classes):
        if num_classes != 1000:
            fc = self.model.classifier[6]
            assert isinstance(fc, nn.Linear)

            in_features = fc.in_features
            new_fc = nn.Linear(in_features, num_classes, bias=True)
            nn.init.normal_(new_fc.weight, 0, 0.01)
            nn.init.zeros_(new_fc.bias)

            self.model.classifier[6] = new_fc

    def forward(self, x):
        res = self.model(x)

        return {KEY_OUTPUT: res}


def get_vggnet(num_classes=1000, arch='vgg16_bn'):
    return VGGNet(num_classes=num_classes, arch=arch)
