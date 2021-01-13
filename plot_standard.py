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

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/standard_description')
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
    #print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

    cell_ids = np.arange(results['events'][0]['tc_filt'].shape[0])
    cell_ids = [x for x in cell_ids if x not in results['excluded_circle']]

    for idx,thresh_level_dict in enumerate(results['events']):

        
        
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
        
        if np.all(np.array(ee) != 0) and False:
           
            tc = ndimage.gaussian_filter(np.load(Path(trial_save,f'{trial_string}_all_tcs.npy')),(0,3))
            tc = tc[cell_ids,:]
            arr =np.concatenate([np.array(x) for x in ee if np.all(x != 0)])
            plt.plot(tc.T)
            plt.show()
            plt.hist(arr[:,1],density = False,histtype = 'step')
            plt.hist(arr[:,2]/arr[:,0],density = False,histtype = 'step')
            plt.show()


            #break
            #raise ValueError('testing')
        
idx = 2
sum_currents = sum_currents[idx]
tot_lengths = tot_lengths[idx]

events = events[idx]
trial =trial[idx]

nonzer_ev = np.concatenate([x for x in events if np.all(x != 0)])


T = 0.2
#plot the event amplitudes
lengths = nonzer_ev[:,0]*T
amplitudes = nonzer_ev[:,1]*100
integrals = nonzer_ev[:,2]*100*T

pos_len = lengths[amplitudes >0]
neg_len = lengths[amplitudes <0]

pos_am = amplitudes[amplitudes >0]
neg_am = amplitudes[amplitudes <0]*-1

pos_int = integrals[integrals > 0]
neg_int = integrals[integrals < 0]*-1

fig,ax = plt.subplots()
ax.plot(pos_len,pos_am,'.',mec = (0,0,0,0),mfc = 'r',alpha = 0.5,markersize = 15,label = 'Positive transients')
ax.plot(neg_len,neg_am,'.',mec = (0,0,0,0),mfc = 'b',alpha = 0.5,markersize = 15, label = 'Negative transients')
#plot the high pass filter approximate effect (use 1000 sample rolling average to remove low freqs in ratio making)

thresh = np.arange(0.002,0.0045,0.0005)[idx]
x_high = plt.gca().get_xlim()[-1]
#ax.plot(np.arange(x_high),np.arange(x_high)/(1000*T) + thresh*100,'--k',linewidth = 3, label = 'Approx. high pass threshold')
ax.set_xlabel('Transient length (s)')
ax.set_ylabel('Absolute Transient Amplitude (%)')
plt.legend(frameon = True,fontsize = 12,loc = (0.5,0.6))
pf.set_all_fontsize(ax, 12)
pf.set_thickaxes(ax, 3)


#also plot the integrals

density = False
cumulative =  False
log = True
split = True
n_bins = 'blocks'

fig1, ax1 = plt.subplots()

linewidth = 3
if split:
    hist = av.hist(pos_int*100, histtype='step', bins=n_bins, density=density,
               cumulative=cumulative, color='r', log=log, linewidth=linewidth, label='Positive')
    hist = av.hist(neg_int*100, histtype='step', bins=n_bins, density=density,
               cumulative=cumulative, color='b', log=log, linewidth=linewidth, label='Negative')
    plt.legend(frameon = False)
else:
    hist = av.hist(integrals*100, histtype='step', bins=n_bins, density=density,
           cumulative=cumulative, color='r', log=log, linewidth=linewidth, label='Integrals')

ax1.set_xlabel('Event integral (% . s) $\propto$ pA ')
ax1.set_ylabel('Log number of observed transients')

pf.set_all_fontsize(ax1, 12)
pf.set_thickaxes(ax1, 3)

#ax1.set_yticks(10.0**(-1*np.arange(5)))

'''
pos_kde = stats.gaussian_kde(np.array([pos_len,pos_am]),bw_method = 10)
#pos_kde.covariance_factor = lambda : 0.5
#pos_kde._compute_covariance()
neg_kde = stats.gaussian_kde(np.array([neg_len,neg_am]),bw_method = 10)



xx,yy = np.meshgrid(np.linspace(0,x_high,200),np.linspace(0,np.max(np.concatenate([pos_am,neg_am])),200))
pos_z = neg_kde(np.array([xx.ravel(),yy.ravel()])).reshape(xx.shape)
plt.pcolormesh(xx,yy,pos_z,shading = 'gouraud')
plt.show()
neg_z = neg_kde(np.array([xx.ravel(),yy.ravel()])).reshape(xx.shape)
plt.pcolormesh(xx,yy,neg_z,shading = 'gouraud')



fig1, ax1 = plt.subplots()

linewidth = 3
if split:
    hist = av.hist(pos_len, histtype='step', bins=n_bins, density=density,
               cumulative=cumulative, color='r', log=log, linewidth=linewidth, label='Positive')
    hist = av.hist(neg_len, histtype='step', bins=n_bins, density=density,
               cumulative=cumulative, color='b', log=log, linewidth=linewidth, label='Negative')
    plt.legend(frameon = False)
else:
    hist = av.hist(lengths, histtype='step', bins=n_bins, density=density,
           cumulative=cumulative, color='r', log=log, linewidth=linewidth, label='Lengths')

ax1.set_xlabel('Event length (s)')
#ax1.set_yticks(10.0**(-1*np.arange(5)))
#plt.legend(frameon = False)



fig1, ax1 = plt.subplots()

linewidth = 3
if split:
    hist = av.hist(pos_am*100, histtype='step', bins=n_bins, density=density,
               cumulative=cumulative, color='r', log=log, linewidth=linewidth, label='Positive')
    hist = av.hist(neg_am*100, histtype='step', bins=n_bins, density=density,
               cumulative=cumulative, color='b', log=log, linewidth=linewidth, label='Negative')
    plt.legend(frameon = False)
else:
    hist = av.hist(amplitudes*100, histtype='step', bins=n_bins, density=density,
           cumulative=cumulative, color='r', log=log, linewidth=linewidth, label='Amplitudes')

ax1.set_xlabel('Event amplitude (%)')
#ax1.set_yticks(10.0**(-1*np.arange(5)))



'''