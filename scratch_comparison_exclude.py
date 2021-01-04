#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 09:27:21 2021

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pyqtgraph as pg
import scipy.ndimage as ndimage


import cancer_functions as canf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)




def plot_events(tc,events,offset = 0,div = 100,color = 'r'):
    plt.plot(tc.T+offset +np.arange(tc.shape[0])/div)
    for idx in events.keys():
        if type(idx) == str:
            continue
        
        for ids in events[idx].T:
            plt.fill_betweenx(np.array([-0.5,0.5])/div+offset+1+idx/div,np.ones(2)*ids[0],np.ones(2)*ids[1],facecolor = color,alpha = 0.2)
    



for idx,data in enumerate(df.itertuples()):
    
    if idx != 34:
        continue
    
    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    
    seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
    
    masks = canf.lab2masks(seg)
    surround_masks = canf.get_surround_masks(masks, dilate = True)
    
    
    surround_tc = np.load(Path(trial_save,f'{trial_string}_all_surround_tcs.npy'))

    ev = canf.detect_events(tc,0.002)
    surrounds_ev = canf.detect_events(surround_tc,0.002)
    
    exc_sur = canf.get_events_exclude_surround_events(tc,surround_tc,max_overlap = 0.1)
    
    plt.cla()
    plot_events(ev['tc_filt'],exc_sur, div = 100,color = 'r')
    plot_events(surrounds_ev['tc_filt'],surrounds_ev, div = 100,color = 'k')
