#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 12:21:15 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import scipy.ndimage as ndimage

import f.plotting_functions as pf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

def get_most_active_traces(num_traces,df,trial_save,trial_string):
    tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    event_dict = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

    idx = 0
    events = event_dict['events'][idx]
    
    keep = [x for x in np.arange(tcs.shape[0])]
    
    #sort by event amounts 
    sort_order = np.array([np.sum(np.abs(events['event_props'][x][:,-1])) if x in events.keys() else 0 for x in range(tcs.shape[0])])
    
    tcs = tcs[keep,:]
    sort_order = np.argsort(sort_order[keep])[::-1]
    
    tcs = tcs[sort_order,:]
    so = np.array(keep)[sort_order]
    
    
    
    tcs = ndimage.gaussian_filter(tcs[:num_traces,...],(0,3))
    so = so[:num_traces]
    
    return tcs,so

ncells = 10
T = 0.2

trial_strings = ['cancer_20201216_slip1_area2_long_acq_long_acq_blue_0.0296_green_0.0765_heated_to_37_1',
                 'cancer_20201216_slip1_area3_long_acq_long_acq_blue_0.0296_green_0.0765_heated_to_37_with_TTX_1']
tcs = []
for t in trial_strings:
    
    print(df[df.trial_string == t].stage)
    tcs.append(get_most_active_traces(ncells,df,Path(save_dir,'ratio_stacks',t), t)[0])
    
    
fig,ax = plt.subplots(ncols = 2)
ax[0].plot(np.arange(tcs[0].shape[1])*T, tcs[0].T + np.arange(ncells)/20, 'k')
ax[1].sharey(ax[0])
ax[1].plot(np.arange(tcs[1].shape[1])*T, tcs[1].T + np.arange(ncells)/20, 'k')

pf.plot_scalebar(ax[0], 0, 0.95, 100, 0.02)
ax[0].axis('off')
ax[1].axis('off')