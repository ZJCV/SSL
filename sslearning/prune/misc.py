# -*- coding: utf-8 -*-

"""
@date: 2021/6/10 下午7:49
@file: misc.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from sslearning.config.key_word import KEY_FILTER, KEY_CHANNEL, KEY_FILTER_AND_CHANNEL


def computer_total(model, dim):
    assert isinstance(model, nn.Module)
    total = 0

    # Count the specified dimension lengths of all Conv
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.shape[dim]

    return total


def computer_weight(weight, prune_way, dimension):
    if prune_way == 'mean_abs':
        return torch.mean(weight.data.abs(), dim=dimension)
    elif prune_way == 'mean':
        return torch.mean(weight.data, dim=dimension)
    elif prune_way == 'sum_abs':
        return torch.sum(weight.data.abs(), dim=dimension)
    elif prune_way == 'sum':
        return torch.sum(weight.data, dim=dimension)
    else:
        raise ValueError(f'{prune_way} does not exists')


def computer_conv(model, conv, index, dim, dimension, prune_way):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.shape[dim]
            conv[index:(index + size)] = computer_weight(m.weight, prune_way, dimension)
            index += size

    return conv, index


def computer_conv_threshold(model, percent, prune_type=KEY_FILTER, prune_way='mean_abs'):
    """
    Calculate pruning threshold of Conv layer
    """
    total = 0

    if prune_type in [KEY_FILTER, KEY_CHANNEL]:
        dim = 0 if prune_type == KEY_FILTER else 1
        dimension = (1, 2, 3) if prune_type == KEY_FILTER else (0, 2, 3)

        total = computer_total(model, dim)

        conv = torch.zeros(total)
        index = 0
        conv, index = computer_conv(model, conv, index, dim, dimension, prune_way)
    elif prune_type == KEY_FILTER_AND_CHANNEL:
        # filter_wise
        total += computer_total(model, 0)
        # channel_wise
        total += computer_total(model, 1)

        conv = torch.zeros(total)
        index = 0
        # filter_wise
        conv, index = computer_conv(model, conv, index, 0, (1, 2, 3), prune_way)
        # channel_wise
        conv, index = computer_conv(model, conv, index, 1, (0, 2, 3), prune_way)
    else:
        raise ValueError(f"{prune_type} does not supports")

    y, i = torch.sort(conv)
    thre_index = int(total * percent)
    thre = y[thre_index]

    return total, thre


# refert to: [Pytorch替换model对象任意层的方法](https://zhuanlan.zhihu.com/p/356273702)
# The core function refers to the implementation of torch.quantification.fuse_modules()
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def set_module_list(model, name_list, module_list, new_module_list):
    for name, module, new_module in zip(name_list, module_list, new_module_list):
        # print(name, module, new_module)
        _set_module(model, name, new_module)


def round_to_multiple_of(val, divisor):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, i.e. (83, 8) -> 88, but (84, 8) -> 88. """
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= val else new_val + divisor
