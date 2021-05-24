#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 19:01:38 2021

@author: peter
"""
import numpy as np
import pandas as pd

from pathlib import Path
HPC = False
top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None

initial_df = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

save_dir = Path(top_dir,'analysis','full')




trial_in = 'cancer_20210119_slip1_area2_long_acq_corr_long_acq_blue_0.0454_green_0.0671_1'

incorrect_cell = 121
correct_value = True


for data in df.itertuples():
    
    if trial_in not in data.trial_string:
        continue
    
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    events = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
    
    events = events['events'][1]