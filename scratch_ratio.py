#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:00:27 2020

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pyqtgraph as pg
import f.general_functions as gf

import scipy.ndimage as ndimage
import skimage.filters

def stack_norm(stack):
    return (stack - stack.min(-1).min(-1)[:,None,None])/(stack.max(-1).max(-1)[:,None,None] - stack.min(-1).min(-1)[:,None,None])


fname = '/home/peter/data/Firefly/cancer/20201028/slip3/cell1/long_acq/ratio_steps_high_both_4x4_50_hz_1/ratio_steps_high_both_4x4_50_hz_1_MMStack_Default.ome.tif'
#fname = '/home/peter/data/Firefly/cancer/20201028/slip2/cell1/steps2_no_seal_test_bright_green/ratio_steps_high_green_green_4x4_25_hz_13/ratio_steps_high_green_green_4x4_25_hz_13_MMStack_Default.ome.tif'
#fname = '/home/peter/data/Firefly/cancer/20201028/slip3/cell2/steps/ratio_steps_high_both_4x4_100_hz_13/ratio_steps_high_both_4x4_100_hz_13_MMStack_Default.ome.tif'
stack = tifffile.imread(fname)

blue = stack[::2,...]
green =  stack[1::2,...]

blue = blue/blue.max()
green = green/green.max()


ratio = blue/green 

tifffile.imsave('./ratio_tst_tif.tif',gf.to_16_bit(ratio))
tifffile.imsave('./green_tst_tif.tif',gf.to_16_bit(green))
tifffile.imsave('./blue_tst_tif.tif',stack[::2,...])


#pg.image(ratio)

im = stack[1,...]