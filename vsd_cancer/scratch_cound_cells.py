#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:09:58 2021

@author: peter
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path





top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20210216_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

mdas = []
mcf10as = []
tgfs = []

for data in df.itertuples():
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))

    if str(data.date)[:4] != '2021':
        mdas.append(seg.max())
    elif 'tgf' in trial_string:
        tgfs.append(seg.max())
    else:
        mcf10as.append(seg.max())

    
    