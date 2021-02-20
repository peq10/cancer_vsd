#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:03:35 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tifffile

from pathlib import Path

from vsd_cancer.functions import cancer_functions as canf
import f.ephys_functions as ef

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20210216_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

LED_fs = 10000
LED_start_offset = 3*10**-3

def time_to_idx(LED_times,cam):
    #for non analog signal objects
    fs = np.mean(np.diff(LED_times))
    start = LED_times[0]
    return np.round((cam - start)/fs).astype(int)


def get_LED_frame_locs(LED,LED_times,cam,T_approx,cam_edge = 'falling'):
    #gets the array index for blue and red frames
    if cam_edge != 'falling':
       raise NotImplementedError('Only implemented for cam falling edge')
    
    #do a rough pass then a second to get LED real value
    ids = time_to_idx(LED_times, [cam[1]+T_approx,cam[1]+3*T_approx,cam[0] - T_approx, cam[0],cam[1] - T_approx, cam[1]])
    zer = LED[ids[0]:ids[1]].mean()
    l1 = LED[ids[2]:ids[3]].mean()
    l2 = LED[ids[4]:ids[5]].mean()
    thr = 0.5*(zer + min(l1,l2)) + zer
    
    LED_thr = LED > thr
    
    ##get actual T
    T = (np.sum(LED_thr.astype(int))/len(cam))/LED_fs
    

        
    
    #now get accurate values 
    ids1 = np.array([time_to_idx(LED_times, cam[::2] - 9*T/10),time_to_idx(LED_times, cam[::2])]).T
    
    ids2 = np.array([time_to_idx(LED_times, cam[1::2] - 4*T/5),time_to_idx(LED_times, cam[1::2] - T/5)]).T
    
    ids3 = np.array([time_to_idx(LED_times, cam[1:-1:2] + T),time_to_idx(LED_times, cam[2::2] - 5*T)]).T

    
    return ids1,ids2,ids3

for data in df.itertuples():
    if data.thresh_use != 'y':
        continue
    
    if data.trial_string != 'cancer_20210122_slip3_area3_long_acq_MCF10A_tgfbeta_long_acq_blue_0.02909_green_0.0672_1':
        continue
    else:
        break
    
    trial_save = Path(save_dir,'ratio_stacks',data.trial_string)
    
    cam =  np.load(Path(trial_save, f'{data.trial_string}_cam.npy'))
    
    LED = np.load(Path(trial_save, f'{data.trial_string}_LED.npy'))
    LED_times = np.arange(len(LED))/LED_fs + cam[0] - LED_start_offset
    
    
    ids = get_LED_frame_locs(LED,LED_times, cam, 3*10**-3)
    blue = np.array([np.median(LED[x[0]:x[1]]) for x in ids[0]])
    green = np.array([np.median(LED[x[0]:x[1]]) for x in ids[1]])
    zer =  np.array([np.median(LED[x[0]:x[1]]) for x in ids[2]])
    #append last as measuring between
    zer = np.append(zer,zer[-1])


    blue -= zer
    green -= zer
    
    
    plt.plot(blue)
    plt.plot(green)
    plt.plot(zer + 0.04)
    plt.show()
    
    print(f'Blue var: {np.std(blue)/np.mean(blue)*100:0.2f}%')
    print(f'Green var: {np.std(green)/np.mean(green)*100:0.2f}%')
    print(data.trial_string)
    
    st = tifffile.imread(data.tif_file)
    
    st_norm = np.copy(st).astype(np.float32)
    st_norm[::2] /= (blue/np.mean(blue))[:,None,None]
    st_norm[1::2] /= (green/np.mean(green))[:,None,None]
    
    tst1 = canf.stack2rat(st)
    tst2 = canf.stack2rat(st_norm)
    