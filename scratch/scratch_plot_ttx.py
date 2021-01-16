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

import scipy.stats as stats

import astropy.visualization as av

from pathlib import Path

import f.plotting_functions as pf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/papers/cancer/v1/TTX_washout/')
if not figsave.is_dir():
    figsave.mkdir()

df = df[(df.use == 'y') & (df.expt == 'TTX_10um_washout')]


trial_string = df.iloc[0].trial_string
n_thresh = len(np.load(Path(Path(save_dir,'ratio_stacks',trial_string),f'{trial_string}_event_properties.npy'),allow_pickle = True).item()['events'])

currents  = [[[],[],[]] for x in range(n_thresh)]
lengths  = [[[],[],[]] for x in range(n_thresh)]

#here we do not differentiate buy slip, just using all cells
for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

    cell_ids = np.arange(results['events'][0]['tc_filt'].shape[0])
    cell_ids = [x for x in cell_ids if x not in results['excluded_circle']]

    for idx,thresh_level_dict in enumerate(results['events']):
    
        
        
        event_props = results['events'][idx]['event_props']
        
        
        if data.stage == 'pre':
            idx2 = 0
        elif data.stage == 'post':
            idx2 = 1
        elif data.stage == 'washout':
            idx2 = 2
        else:
            raise ValueError('oops')
        
        observations = [results['observation_length'][idx][x] for x in cell_ids]
        sum_current = [np.sum(np.abs(event_props[x][:,-1])) if x in event_props.keys() else 0 for x in cell_ids]

        
        currents[idx][idx2].extend(sum_current)
        lengths[idx][idx2].extend(observations)
            

    
pre_current = [np.array(p[0]) for p in currents if len(p[0]) != 0]
pre_length = [np.array(x[0]) for x in lengths]
post_current = [np.array(p[1]) for p in currents if len(p[1]) != 0]
post_length = [np.array(x[1]) for x in lengths]
wash_current = [np.array(p[2]) for p in currents if len(p[2]) != 0]
wash_length = [np.array(x[2]) for x in lengths]

#adjust for observation length
pre_adj = [pre_current[i]/pre_length[i] for i in range(len(pre_current))]
post_adj = [post_current[i]/post_length[i] for i in range(len(post_current))]
wash_adj = [wash_current[i]/wash_length[i] for i in range(len(wash_current))]

#threshold level
idx = 2

#normalise the integrals to 1 max
ma = np.max([pre_adj[idx].max(),post_adj[idx].max(),wash_adj[idx].max()])
pre_adj[idx] /= ma
post_adj[idx] /= ma
wash_adj[idx] /= ma

#first plot the median, IQR of non-zero and proportion of zero current cells

nonzer_curr = [pre_adj[idx][pre_adj[idx]!=0],post_adj[idx][post_adj[idx]!=0],wash_adj[idx][wash_adj[idx]!=0]]
all_curr = [pre_adj[idx],post_adj[idx],wash_adj[idx]]

medians = np.array([np.median(x) for x in nonzer_curr])
IQRs = np.array([[np.percentile(x,25),np.percentile(x,75)] for x in nonzer_curr])

num_zers = np.array([np.sum(pre_adj[idx]==0),np.sum(post_adj[idx]==0),np.sum(wash_adj[idx]==0)])
num_cells_tot = np.array([len(pre_adj[idx]),len(post_adj[idx]),len(wash_adj[idx])])

proportion_zero = num_zers/num_cells_tot


fig1,ax1 = plt.subplots()
ax1.plot(np.arange(3),medians)
ax1.fill_between(np.arange(3),IQRs[:,0],IQRs[:,1],alpha = 0.5)
ax1b = ax1.twinx()
ax1b.plot(np.arange(3),proportion_zero)

#omp = np.array([tot_pre,tot_post,tot_wash])
#comp /= comp.max(axis = 0)
#comp -= comp.mean(axis = 0)


#plt.plot(np.arange(3)[:,None],comp)
#plt.ylim([0,1])



n_bins = 'knuth'
plt.cla()

density = True
cumulative =  True
log = True

fig,ax1 = plt.subplots()

linewidth = 3
pres_hist = av.hist(pre_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative,color = 'r',log = log,linewidth = linewidth,label = 'Pre-TTX')
#pres_hist = av.hist(pre_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False,color = 'r')
post_hist = av.hist(post_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative, color = 'b',log = log,linewidth = linewidth,label = 'TTX 10 $\mathrm{\mu}$M')
#post_hist = av.hist(post_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False, color = 'b')
wash_hist = av.hist(wash_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative,color = 'g',log = log,linewidth = linewidth,label = 'Washout')
#wash_hist = av.hist(wash_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False,color = 'g')
plt.legend(loc = (0.5,0.1),frameon = False,fontsize = 14)
ax1.set_xlabel('Integrated activity activity per cell (a.u.)')
ax1.set_ylabel('Cell fraction')
pf.set_all_fontsize(ax1, 14)
pf.set_thickaxes(ax1, 3)
ax1.tick_params(which = 'minor',width = 3,length = 3)
ax1.set_yticks([0.9,1])
ax1.set_ylim([0.9,1])
ax1.minorticks_off()
fig.savefig()


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