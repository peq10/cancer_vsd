#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:30:21 2021

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

from vsd_cancer.functions import cancer_functions as canf


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
initial_df = Path(top_dir,'analysis','full',f'long_acqs_20210428_experiments_correct_loaded_long.csv')

df = pd.read_csv(initial_df)
roi_df = pd.read_csv(Path(save_dir,'roi_df.csv'))

for data in df.itertuples():
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    if not Path(trial_save,'hand_rois').is_dir():
        continue

    print(data.trial_string)
    im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
    
    tifffile.imsave(Path(trial_save,'hand_rois','im.tif'),gf.to_8_bit(im))