# -*- coding: utf-8 -*-

"""
@date: 2021/6/11 上午10:11
@file: prune_vggnet.py
@author: zj
@description: 
"""

import torch
import warnings

warnings.filterwarnings("ignore")

from sslearning.config.key_word import KEY_DEPTH
from operation import load_model, prune_model, save_model


def prune_depth(cfg_file, prune_way='mean_abs'):
    model, arch_name = load_model(cfg_file, data_shape=(1, 3, 224, 224), device=torch.device('cpu'))

    pruned_type = KEY_DEPTH
    pruned_model, true_pruned_ratio, threshold = prune_model(pruned_type,
                                                             prune_way,
                                                             arch_name,
                                                             model
                                                             )
    print(pruned_model)

    model_name = f'outputs/resnet_pruned_depth/{arch_name}_pruned_{pruned_type}_{prune_way}.pkl'
    save_model(pruned_model, model_name)


if __name__ == '__main__':
    cfg_file = 'configs/resnet/resnet50_cifar100_224_e100_sgd_mslr_ssl_depth_wise_1e_5.yaml'
    prune_depth(cfg_file, prune_way='mean_abs')
