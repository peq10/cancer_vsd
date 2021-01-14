#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:11:07 2021

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import matplotlib.cm

import scipy.ndimage as ndimage

import scipy.stats as stats

import astropy.visualization as av
import astropy.stats as ass

import cancer_functions as canf

from pathlib import Path

import f.plotting_functions as pf
import f.general_functions as gf
import pyqtgraph as pg
import tifffile


top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)


def add_hand_rois(seg,hand_rois):
    
    return 



for idx,data in enumerate(df.itertuples()):
    if '20201216_slip1_area2_long_acq' in data.trial_string:
        break
    
trial_string = data.trial_string
trial_save = Path(save_dir,'ratio_stacks',trial_string)



im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
seg =  np.load(Path(trial_save,f'{trial_string}_seg.npy'))