#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:16:49 2021

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats

import astropy.visualization as av

from pathlib import Path

import f.plotting_functions as pf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/papers/cancer/v1/MCF_V_TGB/')
if not figsave.is_dir():
    figsave.mkdir(parents = True)

df = df[(df.use == 'y') & ((df.expt == 'MCF10A')|(df.expt == 'MCF10A_TGB'))]


currents = []

slips = []

#here we do not differentiate buy slip, just using all cells
for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    
    break