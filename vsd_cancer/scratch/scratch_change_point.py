#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:53:00 2021

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pyqtgraph as pg
import scipy.ndimage as ndimage


import cancer_functions as canf

import ruptures as rpt

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

def get_change_points(t, model_params):
    if model_params is None:
        model = "rank"  # "l2", "rbf"
        min_size = 3
        jump = 5
        penalty = 20
    else:
        model = model_params['model']
        min_size = model_params['min_size']
        jump = model_params['jump']
        penalty = model_params['penalty']
        
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
    result = algo.fit_predict(t,penalty)
    
    return result

for idx, data in enumerate(df.itertuples()):
    if idx != 34:
        continue
    
    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))

    filt_params = {'type':'gaussian','gaussian_sigma':3}
    
    excluded_circle = np.load(Path(trial_save,f'{trial_string}_circle_excluded_rois.npy'))
    #also get circle exclusions
    surround_tc = np.load(Path(trial_save,f'{trial_string}_all_surround_tcs.npy'))
    
    if not np.isnan(data.finish_at):
        observe_to = int(data.finish_at)*5
        tc = tc[:,:observe_to]
        surround_tc = surround_tc[:observe_to]
        
    tc_filt = ndimage.gaussian_filter(tc,(0,3))
    
    #events = {idx: get_change_points(t,{'model':'rbf','min_size':3,'jump':1,'penalty':20}) for t in tc_filt}
    break