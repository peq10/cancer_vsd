#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:32:43 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.gridspec as gridspec

import pandas as pd

from pathlib import Path

import f.plotting_functions as pf

import scipy.stats as stats


top_dir = Path('/home/peter/data/Firefly/cancer')

data_dir = Path(top_dir,'analysis','full')

df = pd.read_csv(Path(data_dir,'all_events_df.csv'))
num_cells_df = pd.read_csv(Path(data_dir,'TTX_active_df.csv'))


#df = df[df.expt == 'standard']


df['exp_stage'] = df.expt + '_' + df.stage
num_cells_df['exp_stage'] = num_cells_df.expt + '_' + num_cells_df.stage

use = [x for x in np.unique(df['exp_stage']) if 'washout' in x]


upper_lim = 6.6
lower_lim = 0
T = 0.2
nbins = 20
only_neg = False
absolute = False
log = True
histtype = 'step'

TTX_level = 10
T = 0.2
dfn = df.copy()
    
    
use_bool = np.array([np.any(x in use) for x in dfn.exp_stage])
dfn = dfn[use_bool]

use_bool2 = np.array([np.any(x in use) for x in num_cells_df.exp_stage])
num_cells_df = num_cells_df[use_bool2]


too_big = np.abs(dfn.event_amplitude) > upper_lim/100
too_small =  np.abs(dfn.event_amplitude) < lower_lim/100
dfn = dfn[np.logical_not(np.logical_or(too_big,too_small))]

if only_neg:
    
    dfn = dfn[dfn['event_amplitude'] < 0]
    
raise ValueError('NEED TO GROUP BY CELL!!!')
amp_bins = np.histogram(dfn['event_amplitude']*100,bins = nbins)[1]
length_bins = np.histogram(dfn['event_length']*T,bins = nbins)[1]

neg = dfn[dfn.stage == 'pre']
pos = dfn[dfn.stage == 'post']
wash = dfn[dfn.stage == 'washout']

