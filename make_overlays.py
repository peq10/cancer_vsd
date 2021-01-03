#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 17:26:21 2021

@author: peter
"""
import numpy as np
from pathlib import Path
import pandas as pd

import scipy.ndimage as ndimage
import scipy.signal as signal

import matplotlib.cm
import tifffile


import f.plotting_functions as pf
import f.general_functions as gf

import cancer_functions as canf

def make_overlay(arr,stack,mask,cmap = matplotlib.cm.hot,alpha_top = 0.7,percent = 5,contrast = [0,0]):
    #mask
    
    overlay = cmap(arr)    
    overlay[...,-1] *= (~mask).astype(int)*alpha_top
    underlay = gf.to_8_bit(matplotlib.cm.Greys_r(stack))
    
    
    final = pf.alpha_composite(gf.to_8_bit(overlay), underlay)
    
    return final

def chunk_overlay(arr,norm_stack,chunk_size,cmap = matplotlib.cm.hot,alpha_top = 0.7,percent = 5, contrast = [0,0]):
    res = np.zeros(arr.shape + (4,),dtype = np.uint8)
    n_chunks,rem = np.divmod(arr.shape[0],chunk_size)  
    mask = np.logical_and(arr < np.percentile(arr,100-percent),arr > np.percentile(arr,percent))
    #apply contrast adjustment
    ma,mi = np.percentile(arr,100-contrast[0]),np.percentile(arr,contrast[0])
    arr[arr > ma] = ma
    arr[arr < mi] = mi
    arr = gf.norm(arr)
    ma2,mi2 = np.percentile(norm_stack,100-contrast[1]),np.percentile(norm_stack,contrast[1])#
    norm_stack[norm_stack > ma2] = ma2
    norm_stack[norm_stack < mi2] = mi2
    norm_stack = gf.norm(norm_stack)
    for i in range(n_chunks):
        res[i*chunk_size:(i+1)*chunk_size,...] =  make_overlay(arr[i*chunk_size:(i+1)*chunk_size,...],
                            norm_stack[i*chunk_size:(i+1)*chunk_size,...],
                            mask[i*chunk_size:(i+1)*chunk_size,...],
                            cmap = cmap,alpha_top=alpha_top,percent = percent,contrast=contrast)
        
    if rem != 0:
        res[-rem:,...] = make_overlay(arr[-rem:,...], norm_stack[-rem:,...],mask[-rem:,...],cmap = cmap,alpha_top=alpha_top,percent = percent,contrast=contrast)
    return res




def make_roi_overlay(events_dict,seg,sz):
    overlay = np.zeros(sz,dtype = int)
    for idx in events_dict.keys():
        for idx2 in range(events_dict[idx].shape[-1]):
            ids = events_dict[idx][:,idx2]
            mask = (seg == idx + 1).astype(int)
            outline = np.logical_xor(mask,ndimage.binary_dilation(mask,iterations = 2)).astype(int)
            overlay[ids[0]:ids[1],...] += outline
    
    overlay = (overlay > 0)

    return overlay




top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)


downsample = 5

for idx,data in enumerate(df.itertuples()):

    
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    if Path(viewing_dir, f'{data.trial_string}_overlay_2.tif').is_file():
        continue



    rat = np.load(Path(trial_save, f'{data.trial_string}_ratio_stack.npy'))[:,2:-2,2:-2]
    rat = ndimage.filters.gaussian_filter(rat,(3,2,2))
    rat = np.pad(rat,((0,0),(2,2),(2,2)),mode = 'edge')
    
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
    filt_params = {'type':'gaussian','gaussian_sigma':3}
    
    exclude_dict = np.load(Path(trial_save,f'{trial_string}_processed_exclusions.npy'),allow_pickle = True).item()
    #add exclusion
    excluded_tc = canf.apply_exclusion(exclude_dict,tc)
    
    events = canf.detect_events(excluded_tc,0.002,filt_params)

    roi_overlay = make_roi_overlay(events,seg,rat.shape)
    exclude_overlay = make_roi_overlay(exclude_dict,seg,rat.shape)
    
    rat = rat[::downsample,...]
    roi_overlay = roi_overlay[::downsample,...]
    exclude_overlay = exclude_overlay[::downsample,...]
    
    
    stack = tifffile.imread(data.tif_file)[::2,...]
    stack = stack[::downsample,...]

    display = chunk_overlay(rat,stack,100,cmap = matplotlib.cm.Spectral,alpha_top=0.2,percent = 50,contrast = [0.5,0.1])
    
    wh = np.where(roi_overlay)
    display[wh[0],wh[1],wh[2],:] = np.array([255,0,0,255])
    
    wh = np.where(exclude_overlay)
    display[wh[0],wh[1],wh[2],:] = np.array([0,128,0,128])
    
    #add overlay for exclusions

    tifffile.imsave(Path(viewing_dir, f'{data.trial_string}_overlay_2.tif'),display)
    
    
