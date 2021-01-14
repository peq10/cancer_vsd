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

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','grey_videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)


downsample = 5

for idx,data in enumerate(df[::-1].itertuples()):

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


    rat2 = np.load(Path(trial_save, f'{data.trial_string}_ratio_stack.npy'))
    rat2 =ndimage.gaussian_filter(rat2,(3,2,2))
    #color balance
    cmin = np.percentile(rat2,0.1)
    cmax = np.percentile(rat2,99.9)
    rat2[np.where(rat2<cmin)] = cmin
    rat2[np.where(rat2>cmax)] = cmax
    tifffile.imsave(Path(viewing_dir, f'{data.trial_string}_overlay_2.tif'),gf.to_8_bit(rat2[::2,...]))
    '''
    rat = rat2[:,2:-2,2:-2]
    rat = ndimage.filters.gaussian_filter(rat,(3,2,2))
    rat = np.pad(rat,((0,0),(2,2),(2,2)),mode = 'edge')[::2,...]
    tifffile.imsave(Path(viewing_dir, f'{data.trial_string}_overlay_2.tif'),gf.to_8_bit(rat))
    '''
