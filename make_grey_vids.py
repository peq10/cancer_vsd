#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:07:19 2021

@author: peter
"""
import numpy as np
from pathlib import Path
import pandas as pd

import scipy.ndimage as ndimage


import tifffile
import time
import f.general_functions as gf 
import prox_tv
import pyqtgraph as pg

import cancer_functions as canf

def make_roi_overlay(events_dict,seg,sz):
    overlay = np.zeros(sz,dtype = int)
    for idx in events_dict.keys():
        if type(idx) == str:
            continue
        for idx2 in range(events_dict[idx].shape[-1]):
            ids = events_dict[idx][:,idx2]
            mask = (seg == idx + 1).astype(int)
            outline = np.logical_xor(mask,ndimage.binary_dilation(mask,iterations = 4)).astype(int)
            overlay[ids[0]:ids[1],...] += outline
    
    overlay = (overlay > 0)

    return overlay


top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','grey_videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)


downsample = 5

for idx,data in enumerate(df.itertuples()):

    t0 = time.time()
    
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    print(trial_string)
    

    if Path(viewing_dir, f'{data.trial_string}_overlay_2.tif').is_file() and False:
        continue

    if data.use == 'n':
        continue
    #if trial_string != 'cancer_20201203_slip1_area2_long_acq_corr_corr_long_acqu_blue_0.0551_green_0.0832_heated_to_37_1':
    #    continue
    
    try:
        finish_at = int(data.finish_at)*5
    except ValueError:
        finish_at = None


    rat2 = np.load(Path(trial_save, f'{data.trial_string}_ratio_stack.npy'))[:finish_at]
    rat2 =ndimage.gaussian_filter(rat2,(3,2,2))
    
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))[:finish_at]
    tc -= tc.mean(-1)[:,None] - 1
     
    seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
    filt_params = {'type':'TV','TV_weight':0.01,'gaussian_sigma':3}
    
    #exclude_dict = np.load(Path(trial_save,f'{trial_string}_processed_exclusions.npy'),allow_pickle = True).item()
    #add exclusion
    #excluded_tc = canf.apply_exclusion(exclude_dict,tc)
    masks = canf.lab2masks(seg)

    
    surround_tc = np.load(Path(trial_save,f'{trial_string}_all_surround_tcs.npy'))[:finish_at]
    surround_tc -= np.mean(surround_tc,-1)[:,None] - 1
    excluded_circle = np.load(Path(trial_save,f'{trial_string}_circle_excluded_rois.npy'))
    
    

    events = canf.get_events_exclude_surround_events(tc,surround_tc,
                                                     detection_thresh = 0.006, 
                                                     surrounds_thresh = 0.002,
                                                     filt_params = filt_params, 
                                                     exclude_first=100, 
                                                     excluded_circle = excluded_circle)
    
    
    roi_overlay = make_roi_overlay(events,seg,rat2.shape)
    #exclude_overlay = make_roi_overlay(events['excluded_events'],seg,rat2.shape)
    #exclude_circle_overlay = make_roi_overlay(events['excluded_circle_events'],seg,rat2.shape)
    downsample = 2
    alpha = 0.65

    
    #color balance
    cmin = np.percentile(rat2,0.1)
    cmax = np.percentile(rat2,99.9)
    rat2[np.where(rat2<cmin)] = cmin
    rat2[np.where(rat2>cmax)] = cmax
    rat2 = gf.norm(rat2)[::downsample]
    #alpha composite
    wh = np.where(roi_overlay[::downsample])
    rat2[wh] = rat2[wh]*(1-alpha) + alpha
    
    tifffile.imsave(Path(viewing_dir, f'{data.trial_string}_overlay_2.tif'),gf.to_8_bit(rat2))

    '''
    rat = rat2[:,2:-2,2:-2]
    rat = ndimage.filters.gaussian_filter(rat,(3,2,2))
    rat = np.pad(rat,((0,0),(2,2),(2,2)),mode = 'edge')[::2,...]
    tifffile.imsave(Path(viewing_dir, f'{data.trial_string}_overlay_2.tif'),gf.to_8_bit(rat))
    '''
