# -*- coding: utf-8 -*-

"""
@date: 2021/7/5 下午10:16
@file: __init__.py.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN
from zcls.config import get_cfg_defaults


def add_custom_config(_C):
    # Add your own customized configs.
    _C.SSL = CN()
    _C.SSL.SPARSITY_REGULARIZATION = False
    # layer_wise
    _C.SSL.TYPE = "filter_and_channel_wise"
    # filter_wise
    _C.SSL.LAMBDA_N = 1e-5
    # channel_wise
    _C.SSL.LAMBDA_C = 1e-5
    # filter_shape_wise
    _C.SSL.LAMBDA_S = 1e-5
    # depth_wise
    _C.SSL.LAMBDA_D = 1e-5

    return _C


cfg = add_custom_config(get_cfg_defaults())
