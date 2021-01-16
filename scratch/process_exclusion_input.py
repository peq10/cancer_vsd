#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:51:17 2021

@author: peter
"""
import numpy as np

import pandas as pd

import scipy.interpolate as interp

import matplotlib.pyplot as plt
from pathlib import Path

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)


for df_idx,data in enumerate(df.itertuples()):
    if df_idx ==4:
        #break
        pass
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    exc_save = Path(trial_save,f'{trial_string}_excluded_times.npy')
    
    
    pts = np.load(exc_save)
    
    if len(pts) == 0:
        np.save(Path(trial_save,f'{trial_string}_processed_exclusions.npy'),{})
        continue
    
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    
    #always input 
    polygons = pts.reshape((-1,4,2))
    
    
    #make top and bottom flat - push to outer boundary and organise
    proc_poly = np.zeros_like(polygons).astype(int)
    for idx, p in enumerate(polygons):
        
        #find which are top and bottom and 'flatten' polygon top and bottom        
        m = np.mean(p[:,1])
        mi = p[:,1].min()
        ma = p[:,1].max()
        p[p[:,1] < m,1] = mi
        p[p[:,1] > m,1] = ma
        p[:,1] = (p[:,1] - 1)*100
        p = np.round(p).astype(int)
        p = p[p.argsort(axis = 0)[:,1],:]
        proc_poly[idx] = p
        
    #now translate into block of specific cells and times
    exclude_dict = {}
    for p in proc_poly:
        
        start_line = interp.interp1d(p[0::2,1],p[0::2,0])
        stop_line = interp.interp1d(p[1::2,1],p[1::2,0])
        mi,ma = p[:,1].min(),p[:,1].max()
        if mi < 0:
            mi = 0
        if ma > tc.shape[0]-1:
            ma = tc.shape[0] - 1
        
        for idx in range(mi,ma+1):
            if idx not in exclude_dict.keys():
                exclude_dict[idx] = [[start_line(idx).astype(int),stop_line(idx).astype(int)]]
            else:
                exclude_dict[idx].append([start_line(idx).astype(int),stop_line(idx).astype(int)])
                
    for key in exclude_dict.keys():
        exclude_dict[key] = np.array(exclude_dict[key]).T 
    
    #save exclude dict
    np.save(Path(trial_save,f'{trial_string}_processed_exclusions.npy'),exclude_dict)
    
    #now display the exclusions


    
    excluded_tc = np.copy(tc)
    
    for roi in exclude_dict.keys():
        for i in range(exclude_dict[roi].shape[-1]):
            ids = exclude_dict[roi][:,i]
            excluded_tc[roi,ids[0]:ids[1]] = 1
            
    plt.cla()
    plt.plot(excluded_tc.T + np.arange(tc.shape[0])/100)
    plt.pause(1)
    #break