#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:48:36 2021

@author: peter
"""

import pandas as pd
from pathlib import Path
import shutil

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing')

df_file = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')
df = pd.read_csv(df_file)
    
hand_roi_dir = Path(top_dir,'Yilin/hand_rois')

for d in hand_roi_dir.iterdir():
    print(d)
    
    trial_string = d.parts[-1][:-1*len('_hand_rois')]
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    shutil.copytree(d,Path(trial_save,'hand_rois'))

'''
for idx,data in enumerate(df.itertuples()):
    if HPC_num is not None: #allows running in parallel on HPC
        if idx != HPC_num:
            continue
        
        
    parts = Path(data.tif_file).parts
    trial_string = '_'.join(parts[parts.index('cancer'):-1])
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    

    
    if Path(trial_save,'hand_rois').is_dir():
        print(trial_save)
        shutil.copytree(Path(trial_save,'hand_rois'),Path(top_dir,'Yilin/hand_rois',f'{trial_string}_hand_rois'))

    else:
        continue
    
'''