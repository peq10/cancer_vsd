#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:19:23 2020

@author: peter
"""

import pandas as pd
import numpy as np
from pathlib import Path

from cellpose import models

import f.general_functions as gf
import scipy.ndimage as ndimage

from vsd_cancer.functions import cancer_functions as canf


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

def segment_cellpose(df_file,save_dir, HPC_num = None):

    df = pd.read_csv(df_file)
    
    ims = []
    savenames = []
    extra_rois = []
    for idx,data in enumerate(df.itertuples()):
        if HPC_num is not None: #allows running in parallel on HPC
            if idx != HPC_num:
                continue
        
        parts = Path(data.tif_file).parts
        trial_string = '_'.join(parts[parts.index('cancer'):-1])
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        if Path(trial_save,'hand_rois').is_dir():
            im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
            extra_rois.append(np.array([gf.read_roi_file(x,im_dims = im.shape)[1] for x in Path(trial_save,'hand_rois').glob('*.roi')]))
        else:
            print('Only doing hand rois! ACHTUNG!')
            continue
        
        im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
        if len(im.shape) > 2:
            im = im[...,0,:,:]
        ims.append(im)
        
        savenames.append(Path(trial_save,f'{trial_string}_seg.npy'))
        

        
    model = models.Cellpose(gpu=False, model_type='cyto')
    masks, flows, styles, diams = model.eval(ims, diameter=30, channels=[0,0])
    
    
    for idx in range(len(ims)):
        if len(extra_rois[idx]) != 0:
            mask = add_hand_rois(masks[idx],extra_rois[idx])
            np.save(savenames[idx],mask)
        else:
            np.save(savenames[idx],masks[idx])