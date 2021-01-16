#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:11:07 2021

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import matplotlib.cm

import scipy.ndimage as ndimage

import scipy.stats as stats

import astropy.visualization as av
import astropy.stats as ass

import cancer_functions as canf

from pathlib import Path

import f.plotting_functions as pf
import f.general_functions as gf
import pyqtgraph as pg
import tifffile


top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)


def add_hand_rois(seg,hand_rois):
    ma = seg.max()
    res = np.copy(seg)
    #add extr rois
    for idx,roi in enumerate(hand_rois):
        wh = np.where(roi)
        res[wh] = ma + idx + 1
    
    #now reorder so they are ordered like before
    masks = canf.lab2masks(res)
    locs = np.array([ndimage.measurements.center_of_mass(x) for x in masks])
    sort = np.argsort(locs[:,0]*seg.shape[1] + locs[:,1])
    
    res = np.sum(masks[sort]*np.arange(1,res.max()+1,dtype = int)[:,None,None],0)
    
    return res



for idx,data in enumerate(df.itertuples()):
    if '20201216_slip1_area2_long_acq' in data.trial_string:
        break
    
trial_string = data.trial_string
trial_save = Path(save_dir,'ratio_stacks',trial_string)

extra_rois = []


im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
seg =  np.load(Path(trial_save,f'{trial_string}_seg.npy'))

if Path(trial_save,'hand_rois').is_dir():
    extra_rois.append(np.array([gf.read_roi_file(x,im_dims = im.shape)[1] for x in Path(trial_save,'hand_rois').glob('*.roi')]))

hand_rois = extra_rois[0]

res = add_hand_rois(seg, hand_rois)