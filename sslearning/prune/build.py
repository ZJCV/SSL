# -*- coding: utf-8 -*-

"""
@date: 2021/7/10 下午2:19
@file: build.py
@author: zj
@description: 
"""

from ..config.key_word import KEY_CHANNEL, KEY_FILTER_AND_CHANNEL, KEY_FILTER

from . import prune_vggnet_by_channel
from . import prune_vggnet_by_filter
from . import prune_vggnet_by_filter_and_channel
from . import prune_resnet_by_depth


def build_prune(model, model_name='vgg',
                ratio=0.2, prune_type=KEY_CHANNEL, prune_way='mean_abs',
                minimum_channels=8, divisor=8):
    if 'vgg' in model_name:
        if prune_type == KEY_CHANNEL:
            return prune_vggnet_by_channel.prune(model, ratio, prune_way=prune_way,
                                                 minimum_channels=minimum_channels, divisor=divisor)
        elif prune_type == KEY_FILTER:
            return prune_vggnet_by_filter.prune(model, ratio, prune_way=prune_way,
                                                minimum_channels=minimum_channels, divisor=divisor)
        elif prune_type == KEY_FILTER_AND_CHANNEL:
            return prune_vggnet_by_filter_and_channel.prune(model, ratio, prune_way=prune_way,
                                                            minimum_channels=minimum_channels, divisor=divisor)
        else:
            raise ValueError(f'{model_name} does not supports')
    elif 'resnet' in model_name:
        return prune_resnet_by_depth.prune(model, prune_way)
    else:
        raise ValueError(f'{model_name} does not supports')
