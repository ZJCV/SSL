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

from sslearning.config.key_word import KEY_CHANNEL, KEY_FILTER, KEY_FILTER_AND_CHANNEL
from operation import load_model, prune_model, save_model


def prune_filter(cfg_file, prune_way='mean_abs', pruned_rate=0.2, minimum_channels=8, divisor=8):
    model, arch_name = load_model(cfg_file, data_shape=(1, 3, 224, 224), device=torch.device('cpu'))

    pruned_type = KEY_FILTER
    pruned_model, true_pruned_ratio, threshold = prune_model(pruned_type,
                                                             prune_way,
                                                             arch_name,
                                                             model,
                                                             ratio=pruned_rate,
                                                             minimum_channels=minimum_channels,
                                                             divisor=divisor,
                                                             )
    print(pruned_model)
    print('pruned ratio:', true_pruned_ratio)
    print('threshold:', threshold)

    model_name = f'outputs/vggnet_pruned_filter/{arch_name}_pruned_{pruned_type}_{prune_way}_{pruned_rate}.pkl'
    save_model(pruned_model, model_name)


def prune_channel(cfg_file, prune_way='mean_abs', pruned_rate=0.2, minimum_channels=8, divisor=8):
    model, arch_name = load_model(cfg_file, data_shape=(1, 3, 224, 224), device=torch.device('cpu'))

    pruned_type = KEY_CHANNEL
    pruned_model, true_pruned_ratio, threshold = prune_model(pruned_type,
                                                             prune_way,
                                                             arch_name,
                                                             model,
                                                             ratio=pruned_rate,
                                                             minimum_channels=minimum_channels,
                                                             divisor=divisor,
                                                             )
    print(pruned_model)
    print('pruned ratio:', true_pruned_ratio)
    print('threshold:', threshold)

    model_name = f'outputs/vggnet_pruned_channel/{arch_name}_pruned_{pruned_type}_{prune_way}_{pruned_rate}.pkl'
    save_model(pruned_model, model_name)


def prune_filter_and_channel(cfg_file, prune_way='mean_abs', pruned_rate=0.2, minimum_channels=8, divisor=8):
    model, arch_name = load_model(cfg_file, data_shape=(1, 3, 224, 224), device=torch.device('cpu'))

    pruned_type = KEY_FILTER_AND_CHANNEL
    pruned_model, true_pruned_ratio, threshold = prune_model(pruned_type,
                                                             prune_way,
                                                             arch_name,
                                                             model,
                                                             ratio=pruned_rate,
                                                             minimum_channels=minimum_channels,
                                                             divisor=divisor,
                                                             )
    print(pruned_model)
    print('pruned ratio:', true_pruned_ratio)
    print('threshold:', threshold)

    model_name = f'outputs/vggnet_pruned_filter_and_channel/{arch_name}_pruned_{pruned_type}_{prune_way}_{pruned_rate}.pkl'
    save_model(pruned_model, model_name)


if __name__ == '__main__':
    # cfg_file = 'configs/vggnet/vgg16_bn_cifar100_224_e100_sgd_mslr_ssl_filter_wise_1e_5.yaml'
    # prune_filter(cfg_file, prune_way='group_lasso', pruned_rate=0.2, minimum_channels=8, divisor=8)
    # prune_filter(cfg_file, prune_way='mean_abs', pruned_rate=0.2, minimum_channels=8, divisor=8)
    # prune_filter(cfg_file, prune_way='mean', pruned_rate=0.2, minimum_channels=8, divisor=8)
    # prune_filter(cfg_file, prune_way='sum_abs', pruned_rate=0.2, minimum_channels=8, divisor=8)
    # prune_filter(cfg_file, prune_way='sum', pruned_rate=0.2, minimum_channels=8, divisor=8)
    # prune_filter(cfg_file, prune_way='mean_abs', pruned_rate=0.4, minimum_channels=8, divisor=8)
    # prune_filter(cfg_file, prune_way='mean_abs', pruned_rate=0.6, minimum_channels=8, divisor=8)

    # cfg_file = 'configs/vggnet/vgg16_bn_cifar100_224_e100_sgd_mslr_ssl_channel_wise_1e_5.yaml'
    # prune_channel(cfg_file, prune_way='group_lasso', pruned_rate=0.2, minimum_channels=8, divisor=8)
    # prune_channel(cfg_file, prune_way='mean_abs', pruned_rate=0.2, minimum_channels=8, divisor=8)
    # prune_channel(cfg_file, prune_way='mean_abs', pruned_rate=0.4, minimum_channels=8, divisor=8)
    # prune_channel(cfg_file, prune_way='mean_abs', pruned_rate=0.6, minimum_channels=8, divisor=8)

    cfg_file = 'configs/vggnet/vgg16_bn_cifar100_224_e100_sgd_mslr_ssl_filter_and_channel_wise_1e_5.yaml'
    prune_filter_and_channel(cfg_file, prune_way='group_lasso', pruned_rate=0.2, minimum_channels=8, divisor=8)
    # prune_filter_and_channel(cfg_file, prune_way='mean_abs', pruned_rate=0.2, minimum_channels=8, divisor=8)
    # prune_filter_and_channel(cfg_file, prune_way='mean_abs', pruned_rate=0.4, minimum_channels=8, divisor=8)
    # prune_filter_and_channel(cfg_file, prune_way='mean_abs', pruned_rate=0.6, minimum_channels=8, divisor=8)
