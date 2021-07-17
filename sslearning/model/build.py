# -*- coding: utf-8 -*-

"""
@date: 2021/6/11 下午2:13
@file: build.py
@author: zj
@description: 
"""

import torch
from zcls.util.checkpoint import CheckPointer
from zcls.util import logging

logger = logging.get_logger(__name__)

from .vggnet import get_vggnet
from .resnet import get_resnet


def build_model(cfg):
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    arch_name = cfg.MODEL.RECOGNIZER.NAME

    if 'pruned' in arch_name:
        path = cfg.MODEL.RECOGNIZER.PRELOADED
        logger.info(f'load pruned model: {path}')
        model = torch.load(path)
    else:
        if 'vgg' in arch_name:
            model = get_vggnet(num_classes=num_classes, arch=arch_name)
        elif 'resnet' in arch_name:
            model = get_resnet(num_classes=num_classes, arch=arch_name)
        else:
            raise ValueError(f"{arch_name} doesn't exists")

        preloaded = cfg.MODEL.RECOGNIZER.PRELOADED
        if preloaded != "":
            logger.info(f'load preloaded: {preloaded}')
            cpu_device = torch.device('cpu')
            check_pointer = CheckPointer(model)
            check_pointer.load(preloaded, map_location=cpu_device)
            logger.info("finish loading model weights")

    return model
