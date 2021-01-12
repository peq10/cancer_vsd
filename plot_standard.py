#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:00:47 2021

@author: peter
"""
#a script to plot the 

#a script to plot the TTX 10 um trials

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.ndimage as ndimage

import scipy.stats as stats

import astropy.visualization as av
import astropy.stats as ass

from pathlib import Path

import f.plotting_functions as pf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/TTX_washout')
if not figsave.is_dir():
    figsave.mkdir(parents = True)

expts_use = ['standard','TTX_10um','TTX_10_um_washout','TTX_1um']
use = [x in expts_use for x in df.expt]

stage_use = ['nan','pre']
use2 = [str(x) in stage_use for x in df.stage]

df = df[(df.use == 'y') & (use) & (use2)]

trial_string = df.iloc[0].trial_string
n_thresh = len(np.load(Path(Path(save_dir,'ratio_stacks',trial_string),f'{trial_string}_event_properties.npy'),allow_pickle = True).item()['events'])

sum_currents  = [[] for x in range(n_thresh)]
tot_lengths  = [[] for x in range(n_thresh)]

events =  [[] for x in range(n_thresh)]

trial = [[] for x in range(n_thresh)]

#here we do not differentiate buy slip, just using all cells
for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

    cell_ids = np.arange(results['events'][0]['tc_filt'].shape[0])
    cell_ids = [x for x in cell_ids if x not in results['excluded_circle']]

    for idx,thresh_level_dict in enumerate(results['events']):
        if idx != 2:
            continue
        
        
        event_props = results['events'][idx]['event_props']
        
        
        #these are on a cell by cell basis
        observations = [results['observation_length'][idx][x] for x in cell_ids]
        sum_current = [np.sum(np.abs(event_props[x][:,-1])) if x in event_props.keys() else 0 for x in cell_ids]
        
        sum_currents[idx].extend(sum_current)
        tot_lengths[idx].extend(observations)
        
        
        #these are on an event by event basis
        ee = [event_props[x] if x in event_props.keys() else np.array([0,0,0])[None,:] for x in cell_ids]
        events[idx].extend(ee)
        trial[idx].extend([data.trial_string for x in cell_ids])
        
        if np.any(np.concatenate(ee)[:,1] >1/100) and False:
            vec = [True if np.any(x[:,1] > 1/100) else False for x in ee]
            ii = np.where(vec)[0][0]
            cell = cell_ids[ii]
            tc = ndimage.gaussian_filter(np.load(Path(trial_save,f'{trial_string}_all_tcs.npy')),(0,3))
            
            eve = ee[ii]
            
            plt.cla()
            plt.plot((tc[cell,:]-1)*100)
            for l in results['events'][idx][cell].T:
                plt.fill_betweenx([(tc[cell]-1).min()*100,(tc[cell]-1).max()*100],l[0],l[1],facecolor = 'r',alpha = 0.5)
            raise ValueError('testing')
        
idx = 2
sum_currents = sum_currents[idx]
tot_lengths = tot_lengths[idx]

events = events[idx]
trial =trial[idx]

nonzer_ev = np.concatenate([x for x in events if np.all(x != 0)])


T = 0.2
#plot the event amplitudes
lengths = nonzer_ev[:,0]*T
amplitudes = nonzer_ev[:,1]
integrals = nonzer_ev[:,2]


density = True
cumulative =  False
log = True
n_bins = 'blocks'

fig1, ax1 = plt.subplots()

linewidth = 3
hist = av.hist(lengths, histtype='step', bins=n_bins, density=density,
               cumulative=cumulative, color='r', log=log, linewidth=linewidth, label='Lengths')

ax1.set_xlabel('Event length (s)')
ax1.set_yticks(10.0**(-1*np.arange(5)))
#plt.legend(frameon = False)



fig1, ax1 = plt.subplots()

linewidth = 3
hist = av.hist(amplitudes*100, histtype='step', bins=n_bins, density=density,
               cumulative=cumulative, color='r', log=log, linewidth=linewidth, label='Amplitudes')

ax1.set_xlabel('Event amplitude (%)')
#ax1.set_yticks(10.0**(-1*np.arange(5)))


fig1, ax1 = plt.subplots()

linewidth = 3
hist = av.hist(integrals, histtype='step', bins=n_bins, density=density,
               cumulative=cumulative, color='r', log=log, linewidth=linewidth, label='Amplitudes')

ax1.set_xlabel('Event integral (%)')
#ax1.set_yticks(10.0**(-1*np.arange(5)))
