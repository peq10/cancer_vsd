#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:26:03 2020

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from pathlib import Path


def lab2masks(seg):
    masks = []
    for i in range(1,seg.max()):
        masks.append((seg == i).astype(int))
    return np.array(masks)

tst = Path('/home/peter/data/Firefly/cancer/analysis/ratio_stacks/cancer_20201117_slip2_area3_long_acq_ratio_slow_scan_blue_0.0255_green_0.0445_v_confluenbt_1')

im = np.load([f for f in Path(tst).glob('*im.npy')][0])
seg = np.load([f for f in Path(tst).glob('*seg.npy')][0])


masks = lab2masks(seg)