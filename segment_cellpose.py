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


def add_hand_rois(seg,hand_rois):
    
    return seg2

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
        im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
        if len(im.shape) > 2:
            im = im[...,0,:,:]
        ims.append(im)
        
        savenames.append(Path(trial_save,f'{trial_string}_seg.npy'))
        
        if Path(trial_save,'hand_rois').is_dir():
            extra_rois.append(np.array([gf.read_roi_file(x,im_dims = im.shape) for x in Path(trial_save,'hand_rois').glob('*.roi')]))

        
        
    model = models.Cellpose(gpu=False, model_type='cyto')
    masks, flows, styles, diams = model.eval(ims, diameter=30, channels=[0,0])
    
    
    for idx in range(len(ims)):
        if len(extra_rois[idx]) != 0:
            mask = add_hand_rois(masks[idx],extra_rois[idx])
            np.save(savenames[idx],masks[idx])
        else:
            np.save(savenames[idx],masks[idx])