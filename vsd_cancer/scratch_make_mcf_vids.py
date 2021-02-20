#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:46:20 2021

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import tifffile



top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20210216_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)


for data in df.itertuples():
    if '20210122_slip1_area3' not in data.trial_string:
        continue
    else:
        print(data.trial_string)
    
    
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    stack = tifffile.imread(data.tif_file)


    green = stack[1::2]
    green = green.astype(float)
    green = (255*(green - green.min())/(green.max() - green.min())).astype(np.uint8)
    tifffile.imsave(Path(top_dir,'analysis','tmp_plotting',f'{trial_string}.tif'),green[::2])