#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:09:07 2021

@author: peter
"""
#a script to get all the events

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




for idx, data in enumerate(df.itertuples()):
    #if idx != 34:
    #    continue
    
    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))

    filt_params = {'type':'gaussian','gaussian_sigma':3}
    
    exclude_dict = np.load(Path(trial_save,f'{trial_string}_processed_exclusions.npy'),allow_pickle = True).item()

    #add exclusion
    excluded_tc = canf.apply_exclusion(exclude_dict,tc)
    
    #also get circle exclusions
    excluded_circle = np.load(Path(trial_save,f'{trial_string}_circle_excluded_rois.npy'))
    
    #apply circle exclusions
    excluded_tc[excluded_circle,:] = 1

    event_props = canf.get_event_properties(excluded_tc,0.002,filt_params,exclude_first = 250) 
    
    result_dict = {'n_cells': tc.shape[0] - len(excluded_circle),
                  'detected_events': event_props,
                  'observation_length': canf.get_observation_length(exclude_dict,tc)
                  }
    
    all_props = np.concatenate([event_props[p] for p in event_props.keys() if 'props' in str(p)])
    
    np.save(Path(trial_save,f'{trial_string}_event_properties.npy'),result_dict)
    