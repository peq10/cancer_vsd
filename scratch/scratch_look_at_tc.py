#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:14:17 2021

@author: peter
"""

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

import pyqtgraph as pg

import scipy.signal as signal
import scipy.ndimage as ndimage
import skimage.filters

import prox_tv

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)




for idx,data in enumerate(df.itertuples()):
    if idx != 66:
        continue
    
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    print(trial_string)
    
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    
    im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
    #plt.imshow(tc)
    #plt.axis('auto')
    plt.cla()
    #plt.plot(tc.T + np.arange(tc.shape[0])/100)
    
    break


if False:
    stack = np.load(Path(trial_save,f'{trial_string}_ratio_stack.npy'))
    pg.image(stack)
    
def tv_filt_tc(tc,w):
    return np.array([prox_tv.tv1_1d(t,w) for t in tc])
    
tc_filt = signal.medfilt(tc,(1,11))
tc_filt = ndimage.gaussian_filter(tc,(0,3))
#tc_filt = tv_filt_tc(tc, 0.01)
#plt.plot(tc_filt.T + np.arange(tc.shape[0])/100)





def soft_threshold(arr,thresh,to = 1):
    #Thresholds towards to value
    res = np.copy(arr)
    wh = np.where(np.abs(arr - to) < thresh)
    wh_pl = np.where(arr - to >= thresh)
    wh_neg = np.where(arr - to <= -1*thresh)
    res[wh] = to
    res[wh_pl] -= thresh
    res[wh_neg] += thresh
    
    return res


def detect_events(tc,med_kernel,thresh):
    tc_filt = signal.medfilt(tc,(1,med_kernel))
    threshed = soft_threshold(tc_filt,thresh)
    
    #return ids of active cells and start/end locations of activity
    active = np.where(np.sum(threshed - 1,-1))[0]
    
    result = {}
    for idx in active:
        t = threshed[idx,:]
        locs = np.diff((np.abs(t -1) != 0).astype(int),prepend = 1,append = 1)
        result[idx] = np.array((np.where(locs == 1)[0],np.where(locs == -1)[0]))

    return result


    
thresh = 0.002
tst = soft_threshold(tc_filt,thresh)
 
plt.cla()
plt.plot(tst.T + np.arange(tc.shape[0])/100)   
#plt.imshow(tst)
#plt.axis('auto')
#plt.cla()
#plt.plot(tc[122,:])
#plt.plot(tst[122,:])

#t_f = prox_tv.tv1_1d(tc[122,:], 0.005)
#plt.plot(soft_threshold(t_f,thresh))

#plt.plot(t_f)
#plt.cla()
#plt.plot(np.sum(tc_filt,0))
