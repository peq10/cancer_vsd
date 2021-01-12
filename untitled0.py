#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:00:47 2021

@author: peter
"""
#a script to plot the 

#a script to plot the TTX 10 um trials

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats

import astropy.visualization as av
import astropy.stats as ass

from pathlib import Path

import f.plotting_functions as pf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/TTX_washout')
if not figsave.is_dir():
    figsave.mkdir(parents = True)

expts_use = ['standard','TTX_10um','TTX_10_um_washout','TTX_1um']
use = [x in expts_use for x in df.expt]

stage_use = ['nan','pre']
use2 = [str(x) in stage_use for x in df.stage]

df = df[(df.use == 'y') & (use) & (use2)]

trial_string = df.iloc[0].trial_string
n_thresh = len(np.load(Path(Path(save_dir,'ratio_stacks',trial_string),f'{trial_string}_event_properties.npy'),allow_pickle = True).item()['events'])

currents  = [[[],[],[]] for x in range(n_thresh)]
lengths  = [[[],[],[]] for x in range(n_thresh)]