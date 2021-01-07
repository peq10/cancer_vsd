#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:21:19 2021

@author: peter
"""
#a script to plot the TTX 10 um trials

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pyqtgraph as pg
import scipy.ndimage as ndimage

import astropy.visualization as av

import cancer_functions as canf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)



df = df[(df.use == 'y') & ((df.expt == 'TTX_10um_washout') | (df.expt == 'TTX_10um'))]


trial_string = df.iloc[0].trial_string
n_thresh = len(np.load(Path(Path(save_dir,'ratio_stacks',trial_string),f'{trial_string}_event_properties.npy'),allow_pickle = True).item()['events'])

currents  = [[[],[],[]] for x in range(n_thresh)]
lengths  = [[[],[],[]] for x in range(n_thresh)]


for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()



    for idx,thresh_level_dict in enumerate(results['events']):
    
        total_observation = np.sum([x for ii,x in enumerate(results['observation_length'][idx]) if ii not in results['excluded_circle']])
        
        event_props = results['events'][idx]['event_props']
        
        
        if data.stage == 'pre':
            idx2 = 0
        elif data.stage == 'post':
            idx2 = 1
        elif data.stage == 'washout':
            idx2 = 2
        else:
            raise ValueError('oops')
        
        try:
            curr = np.concatenate([event_props[x] for x in event_props.keys()])
        except ValueError:
            curr = np.zeros((1,3))
        
        currents[idx][idx2].append(curr)
        lengths[idx][idx2].append(total_observation)
            

    
pre_current = [np.concatenate(p[0]) for p in currents if len(p[0]) != 0]
pre_length = np.array([x[0] for x in lengths])
post_current = [np.concatenate(p[1]) for p in currents if len(p[1]) != 0]
post_length = np.array([x[1] for x in lengths])
wash_current =[np.concatenate(p[2]) for p in currents if len(p[2]) != 0]
wash_length = np.array([x[2] for x in lengths])


#just plot the total activity
tot_pre = np.array([np.sum(np.abs(pre_current[i][:,-1]))/np.sum(pre_length[i]) for i in range(len(pre_current))])
tot_post = np.array([np.sum(np.abs(post_current[i][:,-1]))/np.sum(post_length[i]) for i in range(len(post_current))])
tot_wash = np.array([np.sum(np.abs(wash_current[i][:,-1]))/np.sum(wash_length[i]) for i in range(len(wash_current))])

comp = np.array([tot_pre,tot_post,tot_wash])
comp /= comp.max(axis = 0)
comp -= comp.mean(axis = 0)


plt.plot(np.arange(3)[:,None],comp)
#plt.ylim([0,1])


idx = 19
n_bins = 'knuth'
plt.cla()

density = False
pres_hist = av.hist(pre_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
                     cumulative=True,color = 'r')
#pres_hist = av.hist(pre_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False,color = 'r')
post_hist = av.hist(post_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
                     cumulative=True, color = 'b')
#post_hist = av.hist(post_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False, color = 'b')
wash_hist = av.hist(wash_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
                     cumulative=True,color = 'g')
#wash_hist = av.hist(wash_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False,color = 'g')

'''

#bins = np.histogram_bin_edges(pre_current[idx][:, -1], bins=n_bins)
pres_hist = av.hist(pre_current[idx][:, -1], histtype='step', bins=n_bins, density=True,
                     weights=np.ones(pre_current[idx].shape[0])/np.sum(pre_length[idx]), cumulative=True,color = 'r')
post_hist = av.hist(post_current[idx][:, -1], histtype='step', bins=n_bins, density=True,
                     weights=np.ones(post_current[idx].shape[0])/np.sum(post_length[idx]), cumulative=True, color = 'b')
wash_hist = av.hist(wash_current[idx][:, -1], histtype='step', bins=n_bins, density=True,
                     weights=np.ones(wash_current[idx].shape[0])/np.sum(wash_length[idx]), cumulative=True,color = 'g')




phist = np.histogram(np.abs(pre_current[idx][:, -1]), bins=n_bins,
                     weights=np.ones(pre_current[idx].shape[0])/np.sum(pre_length[idx]))
pohist = np.histogram(np.abs(post_current[idx][:,-1]),bins = n_bins,weights = np.ones(post_current[idx].shape[0])/np.sum(post_length[idx]))
'''