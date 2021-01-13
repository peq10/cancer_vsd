#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:44:41 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.ndimage as ndimage

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

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/standard_description')
if not figsave.is_dir():
    figsave.mkdir(parents = True)

expts_use = ['standard','TTX_10um','TTX_10_um_washout','TTX_1um']
use = [x in expts_use for x in df.expt]

stage_use = ['nan','pre']
use2 = [str(x) in stage_use for x in df.stage]

df = df[(df.use == 'y') & (use) & (use2)]

trial_string = df.iloc[0].trial_string
n_thresh = len(np.load(Path(Path(save_dir,'ratio_stacks',trial_string),f'{trial_string}_event_properties.npy'),allow_pickle = True).item()['events'])

sum_currents  = [[] for x in range(n_thresh)]
tot_lengths  = [[] for x in range(n_thresh)]

events =  [[] for x in range(n_thresh)]

trial = [[] for x in range(n_thresh)]