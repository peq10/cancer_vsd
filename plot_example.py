#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:10:19 2021

@author: peter
"""
import numpy as np
import matplotlib.cm

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pyqtgraph as pg

import catch22

from pathlib import Path

import pandas as pd

import cancer_functions as canf

import f.image_functions as imf

trial_string = 'cancer_20201215_slip2_area1_long_acq_corr_long_acq_blue_0.0296_green_0.0765_heated_to_37_1'


top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/example')
if not figsave.is_dir():
    figsave.mkdir(parents = True)
    
for idx,data in enumerate(df.itertuples()):
    if data.trial_string == trial_string:
        break
    

trial_save = Path(save_dir,'ratio_stacks',trial_string)
    

im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
  
masks = canf.lab2masks(seg)

tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
tcs = ndimage.gaussian_filter(tcs,(0,3))

event_dict = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

exc_circ = event_dict['excluded_circle']
events = event_dict['events'][2]

keep = [x for x in np.arange(tcs.shape[0]) if x not in exc_circ]

#sort by event amounts 
sort_order = np.array([np.sum(np.abs(events['event_props'][x][:,-1])) if x in events.keys() else 0 for x in range(tcs.shape[0])])

tcs = tcs[keep,:]
masks = masks[keep,...]
sort_order = np.argsort(sort_order[keep])[::-1]

tcs = tcs[sort_order,:]
masks = masks[sort_order,:]
so = np.array(keep)[sort_order]

sep = 50
num = 20
cmap = matplotlib.cm.viridis

fig,ax = plt.subplots()
for i in range(num):
    line = ax.plot((tcs[i]-1)*100 + i*100/sep)
    ev = events[so[i]]
    
    for l in ev.T:
        ax.fill_betweenx([(i-0.5)*100/sep,(i+0.5)*100/sep],l[0],l[1],facecolor = line[0].get_c(),alpha = 0.5)


catch22.catch22_all(tcs[i])