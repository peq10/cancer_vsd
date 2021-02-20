#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:07:53 2021

@author: peter
"""

#a test to look at excluding cells based on the range of their stuff 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats

import pyqtgraph as pg

from pathlib import Path

import elephant
import f.general_functions as gf
import scipy.ndimage as ndimage

from vsd_cancer.functions import cancer_functions as canf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

trial_string = '20201215_slip1_area2'

for data in df.itertuples():
    if trial_string in data.trial_string:
        trial_string = data.trial_string
        break
    
trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
rat2 = np.load(Path(trial_save, f'{data.trial_string}_ratio_stack.npy'))

rat2_filt = ndimage.gaussian_filter(rat2,(3,2,2))


seg = np.load(Path(trial_save, f'{data.trial_string}_seg.npy'))



masks = canf.lab2masks(seg)

masks = canf.get_surround_masks_cellfree(masks)
cellfree = seg == 0
cellfree_t = ndimage.gaussian_filter1d(canf.t_course_from_roi(rat2,cellfree),3)
cellfree_t -= cellfree_t.mean()

#roi = seg == 105
roi = seg == 140
b = 10

w = np.where(roi)
test_w = (w[0] - w[0].min() + b,w[1] - w[1].min() + b)
test = rat2[:,w[0].min()-b:w[0].max()+b,w[1].min()-b:w[1].max()+b]
tst2 = test[:,test_w[0],test_w[1]]

mean = np.mean(tst2,-1)
#mean -= mean.mean()

std = np.std(tst2,-1)
#std -= std.mean()

#plt.plot(np.abs(ndimage.gaussian_filter1d(mean,3) -1) > 3*ndimage.gaussian_filter1d(std,3)/np.sqrt(np.sum(roi)))

def func_t_course_from_roi(nd_stack,roi,func ):

    wh = np.where(roi)
    
    return func(nd_stack[...,wh[0],wh[1]],axis = -1)



#try segmenting on the 
tcs = np.array([canf.t_course_from_roi(rat2, roi) for roi in masks])
stds = np.array([canf.std_t_course_from_roi(rat2, roi,True) for roi in masks])
skew = np.array([func_t_course_from_roi(rat2, roi,stats.skew) for roi in masks])
kurtosis = np.array([func_t_course_from_roi(rat2, roi,stats.kurtosis) for roi in masks])

tc_filts = ndimage.gaussian_filter(tcs,(0,3))
std_filts = ndimage.gaussian_filter(stds,(0,3))


events = np.abs(tc_filts - 1) > 7*std_filts
struc = np.zeros((3,5))
struc[1,:] = 1
events2 = ndimage.binary_opening(events,structure = struc,iterations = 2)
events2 = ndimage.binary_closing(events2,structure = struc,iterations = 2)

wh = np.where(events2)
idxs,locs = np.unique(wh[0],return_index=True)
locs = np.append(locs,len(wh[0]))


def recursive_split_locs(seq):
    #splits a sequence into n adjacent sequences
    diff = np.diff(seq)
    if not np.any(diff != 1):
        return [(seq[0],seq[-1])]
    else:
        wh = np.where(diff != 1)[0][0]+1
        return recursive_split_locs(seq[:wh]) + recursive_split_locs(seq[wh:])  

event_dict = {}
for i,idx in enumerate(idxs):
    llocs = wh[1][locs[i]:locs[i+1]]
    split_locs = np.array(recursive_split_locs(llocs))
    event_dict[idx] = split_locs.T
    
    
    
plt.plot(events2)
plt.axis('auto')

