# -*- coding: utf-8 -*-

"""
@date: 2021/7/6 下午9:48
@file: prune_vggnet_by_channel.py
@author: zj
@description: 对卷积层通道维度进行剪枝
1. 不对第一个Conv2d进行剪枝，其输入为固定通道数；
2. 通道剪枝影响的是上游的滤波器维度，所以需要倒序剪枝
"""

import numpy as np
import torch
import torch.nn as nn

from sslearning.config.key_word import KEY_CHANNEL
from .misc import set_module_list, computer_conv_threshold, round_to_multiple_of, computer_weight
from .layers import create_conv2d, create_batchnorm2d


def prune_conv_bn_relu(old_conv2d, old_batchnorm2d, old_relu, conv_threshold, prune_way,
                       out_filters=512, out_idx=None, minimum_channels=8, divisor=8):
    assert isinstance(old_conv2d, nn.Conv2d)
    assert isinstance(old_batchnorm2d, nn.BatchNorm2d)
    assert isinstance(old_relu, nn.ReLU)

    weight_copy = computer_weight(old_conv2d.weight, prune_way, (0, 2, 3))
    # 如果Conv的通道数小于等于最小通道数，则不进行剪枝操作
    if len(weight_copy) <= minimum_channels:
        in_idx = np.arange(minimum_channels)
    else:
        mask = weight_copy.gt(conv_threshold).float()
        # 输入通道剪枝掩码
        in_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if in_idx.size == 1:
            in_idx = np.resize(in_idx, (1,))

        # 如果剪枝后特征长度小于8或者不是8的倍数，那么向上取整到8的倍数
        old_prune_len = len(in_idx)
        new_prune_len = round_to_multiple_of(old_prune_len, divisor)
        if new_prune_len > old_prune_len:
            mask = weight_copy.le(conv_threshold).float()
            tmp_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if tmp_idx.size == 1:
                tmp_idx = np.resize(tmp_idx, (1,))
            res_idx = np.random.choice(tmp_idx, new_prune_len - old_prune_len, replace=False)

            in_idx = np.array(sorted(np.concatenate((in_idx, res_idx))))

    # 输入通道数
    in_channels = len(in_idx)

    # 新建Conv2d/BatchNorm2d/ReLU
    new_conv2d = create_conv2d(old_conv2d, in_channels, out_filters)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_filters)
    new_relu = nn.ReLU(inplace=True)

    new_conv2d.weight.data = old_conv2d.weight.data[:, in_idx.tolist(), :, :].clone()
    if out_idx is not None:
        new_conv2d.weight.data = new_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
        new_conv2d.bias.data = old_conv2d.bias.data[out_idx.tolist()].clone()

        new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[out_idx.tolist()].clone()
        new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[out_idx.tolist()].clone()
        new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[out_idx.tolist()].clone()
        new_batchnorm2d.running_var = old_batchnorm2d.running_var[out_idx.tolist()].clone()

    return new_conv2d, new_batchnorm2d, new_relu, in_channels, in_idx


def prune_features(module_list, conv_threshold, prune_way, minimum_channels=8, divisor=8):
    """
    已知features的模块构成了，逐个采集Conv层计算通道剪枝个数，重新构建
    """
    new_module_list = list()

    out_filters = 512
    out_idx = None
    idx = len(module_list) - 1
    # 不对最开始的Conv层进行通道剪枝
    while idx > 3:
        if isinstance(module_list[idx], nn.ReLU):
            conv2d, batchnorm2d, relu, out_filters, out_idx = prune_conv_bn_relu(module_list[idx - 2],
                                                                                 module_list[idx - 1],
                                                                                 module_list[idx],
                                                                                 conv_threshold,
                                                                                 prune_way,
                                                                                 out_filters=out_filters,
                                                                                 out_idx=out_idx,
                                                                                 minimum_channels=minimum_channels,
                                                                                 divisor=divisor)
            new_module_list.append(relu)
            new_module_list.append(batchnorm2d)
            new_module_list.append(conv2d)
            idx -= 3
        elif isinstance(module_list[idx], nn.MaxPool2d):
            new_module_list.append(module_list[idx])
            idx -= 1

    old_conv2d = module_list[idx - 2]
    old_batchnorm2d = module_list[idx - 1]
    # 新建Conv2d/BatchNorm2d/ReLU
    new_conv2d = create_conv2d(old_conv2d, 3, out_filters)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_filters)
    new_relu = nn.ReLU(inplace=True)

    new_conv2d.weight.data = old_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
    new_conv2d.bias.data = old_conv2d.bias.data[out_idx.tolist()].clone()

    new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[out_idx.tolist()].clone()
    new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[out_idx.tolist()].clone()
    new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[out_idx.tolist()].clone()
    new_batchnorm2d.running_var = old_batchnorm2d.running_var[out_idx.tolist()].clone()

    new_module_list.append(new_relu)
    new_module_list.append(new_batchnorm2d)
    new_module_list.append(new_conv2d)
    idx -= 3
    assert idx == -1

    new_module_list.reverse()
    return new_module_list


def prune(model, percent, prune_way='mean_abs', minimum_channels=8, divisor=8):
    total, threshold = computer_conv_threshold(model, percent, prune_type=KEY_CHANNEL, prune_way=prune_way)
    model = list(model.children())[0]
    # print(model)

    feature_name_list = list()
    feature_module_list = list()
    for name, children in model.named_children():
        # print(name, children)
        if name == 'features':
            # print(list(children))
            for module_name, module in children.named_modules():
                if module_name == '':
                    continue
                feature_name_list.append(f'{name}.{module_name}')
                feature_module_list.append(module)
                # print(name, module_name, module)

    new_module_list = prune_features(feature_module_list,
                                     conv_threshold=threshold,
                                     prune_way=prune_way,
                                     minimum_channels=minimum_channels,
                                     divisor=divisor)
    assert len(new_module_list) == len(feature_module_list) == len(feature_name_list)
    set_module_list(model, feature_name_list, feature_module_list, new_module_list)

    new_total, _ = computer_conv_threshold(model, percent, prune_type=KEY_CHANNEL, prune_way=prune_way)
    return 1 - (1.0 * new_total / total), threshold
