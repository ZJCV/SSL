# -*- coding: utf-8 -*-

"""
@date: 2021/7/6 下午9:48
@file: prune_vggnet_by_channel.py
@author: zj
@description: 层剪枝，每次仅执行一层剪枝操作
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from sslearning.model.prune_bottleneck import PrunedBottleneck
from sslearning.prune.misc import set_module_list


def computer_bottleneck_weight(m, prune_way):
    assert isinstance(m, Bottleneck)

    weight_list = list()
    weight_list.extend(m.conv1.weight.data.reshape(-1))
    weight_list.extend(m.conv2.weight.data.reshape(-1))
    weight_list.extend(m.conv3.weight.data.reshape(-1))

    weight = torch.from_numpy(np.array(weight_list))

    if prune_way == 'mean_abs':
        return torch.mean(weight.abs())
    elif prune_way == 'mean':
        return torch.mean(weight)
    elif prune_way == 'sum_abs':
        return torch.sum(weight.abs())
    elif prune_way == 'sum':
        return torch.sum(weight)
    else:
        raise ValueError(f'{prune_way} does not supports')


def prune_bottleneck(module_list, prune_way):
    """
    """
    weight_list = list()

    # 计算每个Bottleneck对应的权重
    for module in module_list:
        assert isinstance(module, Bottleneck)

        weight_list.append(computer_bottleneck_weight(module, prune_way).cpu().numpy())

    # 计算权重值最小的位置
    idx = np.argmin(weight_list)

    new_module_list = list()
    for i, module in enumerate(module_list):
        if i == idx:
            new_module_list.append(PrunedBottleneck(module))
        else:
            new_module_list.append(module)

    return new_module_list


def prune(model, prune_way):
    model = list(model.children())[0]
    # print(model)

    # 首先计算每个Bottleneck对应的权重大小，然后从中选出值最小的Bottleneck进行剪枝

    # 第一步，统计所有Bottleneck

    bottleneck_name_list = list()
    bottleneck_module_list = list()
    # 逐个处理layer
    for layer_name, layer in list(model.named_children())[4:8]:
        assert isinstance(layer, nn.Sequential)
        # 遍历每个bottleneck
        for submodule_name, submodule in layer.named_children():
            # 统计每个bottleneck的层，进行剪枝操作
            assert isinstance(submodule, Bottleneck)
            bottleneck_name_list.append(f'{layer_name}.{submodule_name}')
            bottleneck_module_list.append(submodule)

    new_module_list = prune_bottleneck(bottleneck_module_list, prune_way)
    assert len(new_module_list) == len(bottleneck_module_list) == len(bottleneck_name_list)
    set_module_list(model, bottleneck_name_list, bottleneck_module_list, new_module_list)

    return None, None
