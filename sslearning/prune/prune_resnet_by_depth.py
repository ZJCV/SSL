# -*- coding: utf-8 -*-

"""
@date: 2021/7/6 下午9:48
@file: prune_vggnet_by_channel.py
@author: zj
@description: Depth Pruning, only one layer pruning operation is performed at a time
Firstly, the weight of each Bottleneck is calculated;
then the Bottleneck with the smallest value is selected for pruning
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from ..model.prune_bottleneck import PrunedBottleneck
from ..prune.misc import set_module_list
from ..util.misc import group_lasso


def computer_bottleneck_weight(m, prune_way):
    assert isinstance(m, Bottleneck)

    if prune_way == 'group_lasso':
        weight = 0

        weight += group_lasso(m.conv1.weight.data)
        weight += group_lasso(m.conv2.weight.data)
        weight += group_lasso(m.conv3.weight.data)
        return weight
    else:
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


def prune_bottleneck(module_list, prune_way, N=1):
    """
    """
    weight_list = list()

    # Calculate the weight corresponding to each Bottleneck
    for module in module_list:
        assert isinstance(module, Bottleneck)

        weight_list.append(computer_bottleneck_weight(module, prune_way).cpu().numpy())

    # Calculate the position with the smallest weight value
    idx_list = np.argsort(weight_list)[:N]

    new_module_list = list()
    for i, module in enumerate(module_list):
        if i in idx_list:
            new_module_list.append(PrunedBottleneck(module))
        else:
            new_module_list.append(module)

    return new_module_list


def prune(model, prune_way, N=1):
    model = list(model.children())[0]
    # print(model)

    # The first step is to count all Bottleneck
    bottleneck_name_list = list()
    bottleneck_module_list = list()
    # Process layer one by one
    for layer_name, layer in list(model.named_children())[4:8]:
        assert isinstance(layer, nn.Sequential)
        # Traverse each bottleneck
        for submodule_name, submodule in layer.named_children():
            # Counting the layers of each bottleneck and pruning
            assert isinstance(submodule, Bottleneck)
            bottleneck_name_list.append(f'{layer_name}.{submodule_name}')
            bottleneck_module_list.append(submodule)

    new_module_list = prune_bottleneck(bottleneck_module_list, prune_way, N=N)
    assert len(new_module_list) == len(bottleneck_module_list) == len(bottleneck_name_list)
    set_module_list(model, bottleneck_name_list, bottleneck_module_list, new_module_list)

    return None, None
