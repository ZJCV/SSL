# -*- coding: utf-8 -*-

"""
@date: 2021/7/6 下午9:48
@file: prune_vggnet_by_channel.py
@author: zj
@description: 对卷积层滤波器维度和通道维度进行剪枝
1. 对当前层的滤波器维度进行剪枝掩码计算时，已经把下一层的通道维度的剪枝掩码计算好了；到了下一层的通道维度，直接使用上一层得到的滤波器剪枝掩码，再进行通道维度剪枝即可。
2. 不对第一个Conv2d进行通道剪枝，其输入为固定通道数；
"""

import numpy as np
import torch
import torch.nn as nn

from sslearning.config.key_word import KEY_FILTER_AND_CHANNEL
from .misc import set_module_list, computer_conv_threshold, round_to_multiple_of, computer_weight
from .layers import create_linear, create_conv2d, create_dropout, create_batchnorm2d


def compute_mask(weight, conv_threshold, prune_way,
                 dim=(1, 2, 3), minimum_channels=8):
    weight_copy = computer_weight(weight, prune_way, dim)
    # minimum_channels，则不进行剪枝操作
    if len(weight_copy) <= minimum_channels:
        mask = torch.ones(weight_copy.shape).gt(0)
    else:
        mask = weight_copy.gt(conv_threshold)

    return mask.byte()


def computer_out_idx(mask, weight,
                     conv_threshold, prune_way, dim=(1, 2, 3),
                     minimum_channels=8, divisor=8):
    # 输出剪枝掩码
    out_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
    if out_idx.size == 1:
        out_idx = np.resize(out_idx, (1,))

    # 如果剪枝后特征长度小于8或者不是8的倍数，那么向上取整到8的倍数
    old_prune_len = len(out_idx)
    new_prune_len = round_to_multiple_of(old_prune_len, divisor)
    if new_prune_len > old_prune_len:
        weight_copy = computer_weight(weight, prune_way, dim)
        mask = weight_copy.le(conv_threshold).float()
        tmp_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if tmp_idx.size == 1:
            tmp_idx = np.resize(tmp_idx, (1,))
        res_idx = np.random.choice(tmp_idx, new_prune_len - old_prune_len, replace=False)

        out_idx = np.array(sorted(np.concatenate((out_idx, res_idx))))

    return out_idx


def prune_conv_bn_relu(old_conv2d_1, old_batchnorm2d, old_relu, old_conv2d_2,
                       conv_threshold, prune_way='mean_abs',
                       in_channels=3, in_idx=None, minimum_channels=8, divisor=8):
    assert isinstance(old_conv2d_1, nn.Conv2d)
    assert isinstance(old_batchnorm2d, nn.BatchNorm2d)
    assert isinstance(old_relu, nn.ReLU)
    if isinstance(old_conv2d_2, nn.MaxPool2d):
        old_conv2d_2 = None

    # 对当前卷积层执行滤波器剪枝
    out_mask_1 = compute_mask(old_conv2d_1.weight,
                              conv_threshold,
                              prune_way,
                              dim=(1, 2, 3),
                              minimum_channels=minimum_channels)

    # 对下一个卷积层执行通道剪枝
    out_mask_2 = torch.ones(out_mask_1.shape).gt(0) if old_conv2d_2 is None else compute_mask(old_conv2d_2.weight,
                                                                                              conv_threshold,
                                                                                              prune_way,
                                                                                              dim=(0, 2, 3),
                                                                                              minimum_channels=minimum_channels)
    out_mask = out_mask_1 | out_mask_2
    out_idx = computer_out_idx(out_mask, old_conv2d_1.weight,
                               conv_threshold, prune_way, dim=(1, 2, 3),
                               minimum_channels=minimum_channels, divisor=divisor)
    # 输出通道数
    out_filters = len(out_idx)

    # 新建Conv2d/BatchNorm2d/ReLU
    new_conv2d = create_conv2d(old_conv2d_1, in_channels, out_filters)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_filters)
    new_relu = nn.ReLU(inplace=True)

    new_conv2d.weight.data = old_conv2d_1.weight.data[out_idx.tolist(), :, :, :].clone()
    new_conv2d.bias.data = old_conv2d_1.bias.data[out_idx.tolist()].clone()
    if in_idx is not None:
        new_conv2d.weight.data = new_conv2d.weight.data[:, in_idx.tolist(), :, :].clone()

    new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[out_idx.tolist()].clone()
    new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[out_idx.tolist()].clone()
    new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[out_idx.tolist()].clone()
    new_batchnorm2d.running_var = old_batchnorm2d.running_var[out_idx.tolist()].clone()

    return new_conv2d, new_batchnorm2d, new_relu, out_filters, out_idx


def prune_features(module_list, conv_threshold, prune_way, minimum_channels=8, divisor=8):
    """
    已知features的模块构成了，逐个采集Conv层计算滤波器剪枝个数，重新构建
    """
    new_module_list = list()
    idx = 0

    in_channels = 3
    in_idx = None
    while idx < len(module_list):
        old_conv2d_2 = None if (idx + 3) > len(module_list) else module_list[idx + 3]
        if isinstance(module_list[idx], nn.Conv2d):
            conv2d, batchnorm2d, relu, in_channels, in_idx = prune_conv_bn_relu(module_list[idx],
                                                                                module_list[idx + 1],
                                                                                module_list[idx + 2],
                                                                                old_conv2d_2,
                                                                                conv_threshold,
                                                                                prune_way=prune_way,
                                                                                in_channels=in_channels,
                                                                                in_idx=in_idx,
                                                                                minimum_channels=minimum_channels,
                                                                                divisor=divisor)
            new_module_list.append(conv2d)
            new_module_list.append(batchnorm2d)
            new_module_list.append(relu)
            idx += 3
        elif isinstance(module_list[idx], nn.MaxPool2d):
            new_module_list.append(module_list[idx])
            idx += 1
    return new_module_list, in_channels, in_idx


def prune_classifier(module_list, in_channels, in_idx):
    """
    对于VGGNet而言，分类器并不包含BN层，所以仅需对输入通道进行调整即可
    :param module_list:
    :param in_channels:
    :return:
    """
    new_module_list = list()

    old_linear = module_list[0]
    assert isinstance(old_linear, nn.Linear)
    new_linear, in_channels = create_linear(old_linear, in_channels)
    new_idx = torch.arange(512 * 7 * 7).reshape(512, 7, 7)[in_idx, :, :].reshape(-1)
    new_linear.weight.data = old_linear.weight.data[:, new_idx].clone()
    new_linear.bias.data = old_linear.bias.data.clone()

    new_module_list.append(new_linear)
    for old_module in module_list[1:]:
        if isinstance(old_module, nn.Linear):
            new_linear, in_channels = create_linear(old_module, in_channels)

            new_linear.weight.data = old_module.weight.data.clone()
            new_linear.bias.data = old_module.bias.data.clone()
            new_module_list.append(new_linear)
        elif isinstance(old_module, nn.ReLU):
            new_module_list.append(nn.ReLU(inplace=True))
        elif isinstance(old_module, nn.Dropout):
            new_dropout = create_dropout(old_module)
            new_module_list.append(new_dropout)

    return new_module_list


def prune(model, percent, prune_way='mean_abs', minimum_channels=8, divisor=8):
    total, threshold = computer_conv_threshold(model, percent, prune_type=KEY_FILTER_AND_CHANNEL, prune_way=prune_way)

    model = list(model.children())[0]
    # print(model)

    feature_name_list = list()
    feature_module_list = list()
    classifier_name_list = list()
    classifier_module_list = list()
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
        elif name == 'classifier':
            for module_name, module in children.named_modules():
                if module_name == '':
                    continue
                classifier_name_list.append(f'{name}.{module_name}')
                classifier_module_list.append(module)
                # print(name, module_name, module)

    new_module_list, in_channels, in_idx = prune_features(feature_module_list,
                                                          conv_threshold=threshold,
                                                          prune_way=prune_way,
                                                          minimum_channels=minimum_channels,
                                                          divisor=divisor)
    assert len(new_module_list) == len(feature_module_list) == len(feature_name_list)
    set_module_list(model, feature_name_list, feature_module_list, new_module_list)

    new_module_list = prune_classifier(classifier_module_list, in_channels * 7 * 7, in_idx)
    assert len(new_module_list) == len(classifier_module_list) == len(classifier_name_list)
    set_module_list(model, classifier_name_list, classifier_module_list, new_module_list)

    new_total, _ = computer_conv_threshold(model, percent, prune_type=KEY_FILTER_AND_CHANNEL, prune_way=prune_way)
    return 1 - (1.0 * new_total / total), threshold
