# -*- coding: utf-8 -*-

"""
@date: 2021/7/16 下午7:19
@file: operation.py
@author: zj
@description: 
"""

import os
import torch

from zcls.config.key_word import KEY_OUTPUT

from sslearning.config import cfg
from sslearning.model.build import build_model
from sslearning.prune.build import build_prune
from sslearning.util.profile import computer_flops_and_params, compute_model_time


def load_model(config_file, data_shape=(1, 3, 224, 224), device=torch.device('cpu')):
    cfg.merge_from_file(config_file)

    model = build_model(cfg).to(device)
    # print(model)

    computer_flops_and_params(model)
    compute_model_time(data_shape, model, device)
    return model, cfg.MODEL.RECOGNIZER.NAME


def prune_model(prune_type, prune_way, arch_name, model, ratio=0.2, minimum_channels=8, divisor=8):
    pruned_ratio, threshold = build_prune(model, model_name=arch_name,
                                          ratio=ratio, prune_type=prune_type, prune_way=prune_way,
                                          minimum_channels=minimum_channels, divisor=divisor
                                          )
    computer_flops_and_params(model)
    compute_model_time((1, 3, 224, 224), model, torch.device('cpu'))

    return model, pruned_ratio, threshold


def save_model(model, model_name):
    data = torch.randn(1, 3, 224, 224)
    res = model(data)[KEY_OUTPUT]
    print(res.shape)

    output_dir = os.path.split(os.path.abspath(model_name))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model, model_name)
