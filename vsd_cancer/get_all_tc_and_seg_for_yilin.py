#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 08:13:17 2021

@author: peter
"""



import numpy as np

import pandas as pd
from pathlib import Path


top_dir = Path('/home/peter/data/Firefly/cancer')

save_dir = Path(top_dir,'analysis','full')
yilin_dir = Path(top_dir,'Yilin','all_tcs')

initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')

df = pd.read_csv(initial_df)


for idx, data in enumerate(df.itertuples()):

    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    
    
    tc = np.load(Path(trial_save,f'{trial_string}_all_eroded_median_tcs.npy'))
    tc -= np.mean(tc,-1)[:,None] - 1
    
    old_tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    
    #remove end if its gone worng
    if not np.isnan(data.finish_at):
        observe_to = int(data.finish_at)*5
        tc = tc[:,:observe_to]
        old_tc = tc[:,:observe_to]


    seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
    
        np.save(Path(yilin_dir,data.use,f'{trial_string}_all_eroded_median_tcs.npy'),tc)
    np.save(Path(yilin_dir,data.use,f'{trial_string}_all_tcs.npy'),old_tc)
    
    np.save(Path(yilin_dir,data.use,f'{trial_string}_seg.npy'),seg)
    
    
    df.to_csv(Path(yilin_dir,data.use,'long_acqs_20210428_experiments_correct.csv'))
    
    