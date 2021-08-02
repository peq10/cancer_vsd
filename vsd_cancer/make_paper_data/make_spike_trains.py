#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:01:38 2021

@author: peter
"""
import pandas as pd
from pathlib import Path

import numpy as np


def export_spike_trains(save_dir,T = 0.2,only_neg = True):
    df = pd.read_csv(Path(save_dir,'all_events_df.csv'))


    if only_neg:
        df = df[df.event_amplitude < 0]
        
    df['exp_stage'] = df.expt + '_' + df.stage

    #only look at 231s
    use = ['TTX_10um_washout_pre','TTX_1um_pre','TTX_10um_pre','standard_none']
    
    use_bool = np.array([np.any(x in use) for x in df.exp_stage])
    df = df[use_bool]
    
        
        
    trials = np.unique(df.trial_string)
    
    all_trains = []
    for t in trials:
        dft = df[df.trial_string == t]
        
        cells = np.unique(dft['cell_id'])
        
        trial_spikes = {}
        
        for c in cells:
           times = np.array((dft[dft.cell_id == c].event_time)*T)
           pos = np.array([dft[dft.cell_id == c].cell_y.values[0],dft[dft.cell_id == c].cell_x.values[0]])
           trial_spikes[c] = (times,pos)
           
        all_trains.append(trial_spikes)  
        
        
    np.save(Path(save_dir,'all_spike_trains.npy'),all_trains)