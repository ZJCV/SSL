# -*- coding: utf-8 -*-

"""
@date: 2021/6/11 上午10:43
@file: profile.py
@author: zj
@description: 
"""

import time
import torch
from thop import profile


def computer_flops_and_params(model, data_shape=(1, 3, 224, 224)):
    input = torch.randn(data_shape)
    flops, params = profile(model, inputs=(input,), verbose=False)

    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))


def compute_model_time(data_shape, model, device):
    model = model.to(device)
    model.eval()

    t1 = 0.0
    num = 100
    begin = time.time()
    for i in range(num):
        time.sleep(0.01)
        data = torch.randn(data_shape)
        start = time.time()
        model(data.to(device=device, non_blocking=True))
        if i > num // 2:
            t1 += time.time() - start
    t2 = time.time() - begin
    print(f'one process need {t2 / num:.3f}s, model compute need: {t1 / (num // 2):.3f}s')
    model.train()
