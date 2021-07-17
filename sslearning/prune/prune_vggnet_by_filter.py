# -*- coding: utf-8 -*-

"""
@date: 2021/7/6 下午9:48
@file: prune_vggnet_by_channel.py
@author: zj
@description: 
"""

import numpy as np
import torch
import torch.nn as nn

from sslearning.config.key_word import KEY_FILTER
from .misc import set_module_list, computer_conv_threshold, round_to_multiple_of, computer_weight
from .layers import create_linear, create_conv2d, create_dropout, create_batchnorm2d


def prune_conv_bn_relu(old_conv2d, old_batchnorm2d, old_relu, conv_threshold, prune_way,
                       in_channels=3, in_idx=None, minimum_channels=8, divisor=8):
    assert isinstance(old_conv2d, nn.Conv2d)
    assert isinstance(old_batchnorm2d, nn.BatchNorm2d)
    assert isinstance(old_relu, nn.ReLU)

    weight_copy = computer_weight(old_conv2d.weight, prune_way, (1, 2, 3))
    # If the number of channels of BN is less than or equal to minimum_channels, pruning is not performed
    if len(weight_copy) <= minimum_channels:
        out_idx = np.arange(minimum_channels)
    else:
        mask = weight_copy.gt(conv_threshold).float()
        # Output pruning mask
        out_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if out_idx.size == 1:
            out_idx = np.resize(out_idx, (1,))

        # If the feature length after pruning is less than minimum_channels or not a multiple of divisor, it will be rounded up to a multiple of divisor
        old_prune_len = len(out_idx)
        new_prune_len = round_to_multiple_of(old_prune_len, divisor)
        if new_prune_len > old_prune_len:
            mask = weight_copy.le(conv_threshold).float()
            tmp_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if tmp_idx.size == 1:
                tmp_idx = np.resize(tmp_idx, (1,))
            res_idx = np.random.choice(tmp_idx, new_prune_len - old_prune_len, replace=False)

            out_idx = np.array(sorted(np.concatenate((out_idx, res_idx))))

    # Number of output channels
    out_filters = len(out_idx)

    # New Conv2d/BatchNorm2d/ReLU
    new_conv2d = create_conv2d(old_conv2d, in_channels, out_filters)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_filters)
    new_relu = nn.ReLU(inplace=True)

    new_conv2d.weight.data = old_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
    new_conv2d.bias.data = old_conv2d.bias.data[out_idx.tolist()].clone()
    if in_idx is not None:
        new_conv2d.weight.data = new_conv2d.weight.data[:, in_idx.tolist(), :, :].clone()

    new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[out_idx.tolist()].clone()
    new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[out_idx.tolist()].clone()
    new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[out_idx.tolist()].clone()
    new_batchnorm2d.running_var = old_batchnorm2d.running_var[out_idx.tolist()].clone()

    return new_conv2d, new_batchnorm2d, new_relu, out_filters, out_idx


def prune_features(module_list, conv_threshold, prune_way, minimum_channels=8, divisor=8):
    """
    Given the module composition of features, collect Conv layers one by one, calculate the number of pruning filters, and reconstruct them
    """
    new_module_list = list()
    idx = 0

    in_channels = 3
    in_idx = None
    while idx < len(module_list):
        if isinstance(module_list[idx], nn.Conv2d):
            conv2d, batchnorm2d, relu, in_channels, in_idx = prune_conv_bn_relu(module_list[idx],
                                                                                module_list[idx + 1],
                                                                                module_list[idx + 2],
                                                                                conv_threshold,
                                                                                prune_way,
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
    For VGGNet, the classifier does not contain BN layer, so it only needs to adjust the input channel
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
    total, threshold = computer_conv_threshold(model, percent, prune_type=KEY_FILTER, prune_way=prune_way)

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

    new_total, _ = computer_conv_threshold(model, percent, prune_type=KEY_FILTER, prune_way=prune_way)
    return 1 - (1.0 * new_total / total), threshold
