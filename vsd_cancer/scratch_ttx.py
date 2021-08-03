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

df = pd.read_csv(Path(data_dir,'TTX_active_df_by_cell.csv'))



#df = df[df.expt == 'standard']

T = 0.2


df['exp_stage'] = df.expt + '_' + df.stage

df['event_rate'] = (df['n_neg_events'] +  df['n_pos_events'])/(df['obs_length']*T)
df['neg_event_rate'] = (df['n_neg_events'] )/(df['obs_length']*T)

df['integ_rate'] = (df['integrated_events'])/(df['obs_length']*T)
df['neg_integ_rate'] = (df['neg_integrated_events'] )/(df['obs_length']*T)


use = [x for x in np.unique(df['exp_stage']) if 'washout' in x]


dfn = df.copy()
#dfn = dfn[dfn.event_rate < 0.03]
    
    
use_bool = np.array([np.any(x in use) for x in dfn.exp_stage])
dfn = dfn[use_bool]



pre = dfn[dfn.stage == 'pre']
post = dfn[dfn.stage == 'post']
wash = dfn[dfn.stage == 'washout']

nbins = 20
key = 'integ_rate'
log = True
histtype = 'step'
bins = np.histogram(dfn[key],bins = nbins)[1]

plt.figure()
plt.hist(pre[key],bins = bins,log = log,histtype = histtype)
plt.figure()
plt.hist(post[key],bins = bins,log = log,histtype = histtype)
plt.figure()
plt.hist(wash[key],bins = bins,log = log,histtype = histtype)

plt.figure()

plt.plot([np.mean(pre[key]),np.mean(post[key]),np.mean(wash[key])])