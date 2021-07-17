# -*- coding: utf-8 -*-

"""
@date: 2021/6/11 上午10:05
@file: layers.py
@author: zj
@description: 
"""

import torch.nn as nn


def create_conv2d(old_conv2d, in_channels, out_filters, old_groups=None):
    assert isinstance(old_conv2d, nn.Conv2d)
    kernel_size = old_conv2d.kernel_size
    stride = old_conv2d.stride
    padding = old_conv2d.padding
    padding_mode = old_conv2d.padding_mode
    groups = old_groups if old_groups is not None else old_conv2d.groups
    dilation = old_conv2d.dilation
    bias = old_conv2d.bias is not None

    new_conv2d = nn.Conv2d(in_channels,
                           out_filters,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           padding_mode=padding_mode,
                           groups=groups,
                           dilation=dilation,
                           bias=bias
                           )
    return new_conv2d


def create_batchnorm2d(old_batchnorm2d, in_channels):
    assert isinstance(old_batchnorm2d, nn.BatchNorm2d)

    eps = old_batchnorm2d.eps
    momentum = old_batchnorm2d.momentum

    return nn.BatchNorm2d(in_channels, eps=eps, momentum=momentum)


def create_linear(old_linear, in_channels):
    assert isinstance(old_linear, nn.Linear)

    out_channels = old_linear.out_features
    bias = old_linear.bias is not None

    return nn.Linear(in_channels, out_channels, bias=bias), out_channels


def create_dropout(old_dropout):
    assert isinstance(old_dropout, nn.Dropout)

    p = old_dropout.p
    inplace = old_dropout.inplace

    return nn.Dropout(p=p, inplace=inplace)
