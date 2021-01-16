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
import pyqtgraph as pg


import f.general_functions as gf
    
def lab2masks(seg):
    masks = []
    for i in range(1,seg.max()):
        masks.append((seg == i).astype(int))
    return np.array(masks)


def make_all_tc(df_file,save_dir):
    df = pd.read_csv(df_file)
    
    for idx,data in enumerate(df.itertuples()):
        
        parts = Path(data.tif_file).parts
        trial_string = '_'.join(parts[parts.index('cancer'):-1])
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
        
        masks = lab2masks(seg)
        
        stack = np.load(Path(trial_save,f'{trial_string}_ratio_stack.npy'))
        
        tc = np.array([gf.t_course_from_roi(stack,mask) for mask in masks])
        
        np.save(Path(trial_save,f'{trial_string}_all_tcs.npy'),tc)
        
        
tst = Path('/home/peter/data/Firefly/cancer/analysis/ratio_stacks/cancer_20201117_slip2_area3_long_acq_ratio_slow_scan_blue_0.0255_green_0.0445_v_confluenbt_1')

im = np.load([f for f in Path(tst).glob('*im.npy')][0])
seg = np.load([f for f in Path(tst).glob('*seg.npy')][0])


masks = lab2masks(seg)

stack = np.load([f for f in Path(tst).glob('*ratio_stack.npy')][0])



tc = np.array([gf.t_course_from_roi(stack,mask) for mask in masks])

tc_fil = ndimage.median_filter(tc,(1,21))

std = np.std(tc_fil[:,750:-750],-1)

arr_idx = std.argsort()