#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 12:56:29 2021

@author: peter
"""
from pathlib import Path
import pandas as pd

import shutil

redo = False

home = Path.home()

HPC = False
top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None
yilin_save = False
yilins_computer = False
njobs = 10


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing')


df = pd.read_csv(Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv'))

to_dir = Path(save_dir,'yilin_data/tcs_to_yilin')

for data in df.itertuples():
    t = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',t)
    
    all_tcs = Path(trial_save).glob('*tcs.npy')
    
    for seg in all_tcs:
       
        to_seg = Path(to_dir,seg.stem + '.npy')
    
    
        print(seg)
        print(to_seg)
        shutil.copyfile(seg,to_seg)
        
        
    events = Path(trial_save,f'{t}_event_properties.npy')
    to_events = Path(to_dir,events.stem + '.npy')
    shutil.copyfile(events,to_events)
