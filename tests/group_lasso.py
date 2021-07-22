# -*- coding: utf-8 -*-

"""
@date: 2021/7/21 下午9:49
@file: group_lasso.py
@author: zj
@description: 
"""

import torch


def group_lasso(param_group):
    return torch.sum(param_group ** 2)


def group_lasso_by_filter_or_channel(param_group, dimension):
    return torch.sum(param_group ** 2, dim=dimension)


def test_group_lasso():
    data = torch.randn(2, 3, 1, 1)

    res_filter = 0
    # Group LASSO over filters of current layer
    for filter_idx in range(2):
        res_filter += group_lasso(data[filter_idx, :, :, :])

    res_filter_2 = torch.sum(group_lasso_by_filter_or_channel(data, (1, 2, 3)))
    assert res_filter == res_filter_2


if __name__ == '__main__':
    test_group_lasso()
