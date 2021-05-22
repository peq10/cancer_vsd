#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 12:14:06 2021

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.spatial as spatial

import pyqtgraph as pg

from pathlib import Path

#import elephant.unitary_event_analysis as ue

from vsd_cancer.functions import cancer_functions as canf
import quantities as pq
#import neo.core

def skewness_from_roi(nd_stack,roi):
    if len(roi.shape) != 2:
        raise NotImplementedError('Only works for 2d ROIs')
    wh = np.where(roi)
    vals = nd_stack[...,wh[0],wh[1]]
    mean = np.mean(vals,-1)
    std = np.std(vals,-1)
    skew = (np.sum((vals - mean[:,None])**3,-1)/vals.shape[-1])/std
    return skew


top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

trial_string = '20201215_slip2_area1_long_acq_corr'

for data in df.itertuples():
    if trial_string in data.trial_string:
        trial_string = data.trial_string
        break
    
trial_save = Path(save_dir,'ratio_stacks',trial_string)

rat = np.load(Path(trial_save, f'{data.trial_string}_ratio_stack.npy'))
seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
        
masks = canf.lab2masks(seg)

rat2 = ndimage.gaussian_filter(rat,(3,2,2))

tc = np.array([canf.t_course_from_roi(rat,mask) for mask in masks])

skew_tc = np.array([skewness_from_roi(rat, mask) for mask in masks])

tc_filt = ndimage.gaussian_filter(tc,(0,3))


