#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:07:20 2021

@author: peter
"""

#a script to do exclusions of blocks of time


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import cancer_functions as canf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)



def get_exclusion(tc):
    fig,ax = plt.subplots()
    fig.suptitle('Left click to add point, right click to remove. Center to end.\nExclude sections with 4 points.')
    ax.plot(tc.T + np.arange(tc.shape[0])/100)
    fig.canvas.manager.window.showMaximized()
    fig.canvas.manager.window.raise_()
    pts = np.asarray(plt.ginput(-1, timeout=-1))
    plt.close(fig.number)
    return pts

for idx,data in enumerate(df.itertuples()):
    if idx != 20:
        continue
        
    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    exc_save = Path(trial_save,f'{trial_string}_excluded_times.npy')
    
    #if trial_string != 'cancer_20201207_slip2_area1_long_acq_long_acqu_blue_0.03465_green_0.07063_heated_to_37_1':
    #    continue
    
    if exc_save.is_file() and False:
        continue
    

    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    
    if Path(trial_save,f'{trial_string}_processed_exclusions.npy').is_file():
        excluded = np.load(Path(trial_save,f'{trial_string}_processed_exclusions.npy'),allow_pickle = True).item()
    
    excluded_tc = canf.apply_exclusion(excluded, tc)
    
    if True:
        prev_pts = np.load(exc_save)
        pts = get_exclusion(excluded_tc)
        if len(prev_pts) != 0:
            pts = np.concatenate((prev_pts,pts),axis = 0)
        np.save(exc_save,pts)
    else:
        pts = get_exclusion(excluded_tc)
        np.save(exc_save,pts)

