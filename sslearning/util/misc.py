# -*- coding: utf-8 -*-

"""
@date: 2021/7/5 下午10:46
@file: misc.py
@author: zj
@description: 
"""

import torch

from torchvision.models.resnet import Bottleneck

from sslearning.config.key_word import KEY_CHANNEL, KEY_FILTER, KEY_FILTER_AND_CHANNEL, KEY_DEPTH, KEY_FILTER_SHAPE


def group_lasso(param_group):
    return torch.sum(param_group ** 2)


def ssl_loss(model, model_type='resnet', loss_type=KEY_FILTER,
             lambda_n=1e-5, lambda_c=1e-5, lambda_s=1e-5, lambda_d=1e-5):
    assert isinstance(model, torch.nn.Module)
    ssl_loss = 0

    if loss_type in [KEY_FILTER, KEY_CHANNEL, KEY_FILTER_AND_CHANNEL]:
        # Iterate over every layer
        params = list(model.parameters())
        for param in params:
            # Ignore linear or bias parameters
            if len(param.size()) != 4:
                continue

            num_filters, num_channels = param.size()[0], param.size()[1]
            height = param.size()[2]
            width = param.size()[3]

            if loss_type == KEY_FILTER:
                # Group LASSO over filters of current layer
                for filter_idx in range(num_filters):
                    ssl_loss += lambda_n * group_lasso(param[filter_idx, :, :, :])
            elif loss_type == KEY_CHANNEL:
                # Group LASSO over channels of current layer
                for channel_idx in range(num_channels):
                    ssl_loss += lambda_c * group_lasso(param[:, channel_idx, :, :])
            elif loss_type == KEY_FILTER_AND_CHANNEL:
                # Group LASSO over filters of current layer
                for filter_idx in range(num_filters):
                    ssl_loss += lambda_n * group_lasso(param[filter_idx, :, :, :])

                # Group LASSO over channels of current layer
                for channel_idx in range(num_channels):
                    ssl_loss += lambda_c * group_lasso(param[:, channel_idx, :, :])
            elif loss_type == KEY_FILTER_SHAPE:
                # Group LASSO over shapes
                for channel_idx in range(num_channels):
                    for height_idx in range(height):
                        for width_idx in range(width):
                            ssl_loss += lambda_s * group_lasso(param[:, channel_idx, height_idx, width_idx])
            else:
                raise ValueError(f"{type} doesn't supports")
    elif loss_type == KEY_DEPTH:
        if model_type == 'resnet':
            for name, module in model.named_modules():
                if isinstance(module, Bottleneck):
                    ssl_loss += lambda_d * group_lasso(module.conv1.weight.data)
                    ssl_loss += lambda_d * group_lasso(module.conv2.weight.data)
                    ssl_loss += lambda_d * group_lasso(module.conv3.weight.data)
    else:
        raise ValueError(f'{model_type} does not supports')

    return ssl_loss
