#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 14:59:15 2021

@author: peter
"""
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd

import scipy.ndimage as ndimage
import matplotlib

from vsd_cancer.functions import cancer_functions as canf
from vsd_cancer.functions import correlation_functions as corrf

import f.plotting_functions as pf



trial_string_use = 'cancer_20201207_slip1_area1_long_acq_corr_corr_long_acqu_blue_0.03465_green_0.07063_heated_to_37_1'
top_dir = Path('/home/peter/data/Firefly/cancer')
save_dir = Path(top_dir,'analysis','full')
figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
df = pd.read_csv(initial_df)


num_traces = 15
sep = 25

traces = np.array([0,2,3,4,6,8])


T = 0.2

im_scalebar_length_um = 100

for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    #print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    if trial_string != trial_string_use:
        continue
    else:
        break

im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
  
masks = canf.lab2masks(seg)

tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))

event_dict = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

idx = 0
events = event_dict['events'][idx]
event_props = events['event_props']

keep = [x for x in np.arange(tcs.shape[0])]

#sort by event amounts 
sort_order = np.array([np.sum(np.abs(events['event_props'][x][:,-1])) if x in events.keys() else 0 for x in range(tcs.shape[0])])

tcs = tcs[keep,:]
masks = masks[keep,...]
sort_order = np.argsort(sort_order[keep])[::-1]

tcs = tcs[sort_order,:]
masks = masks[sort_order,:]
so = np.array(keep)[sort_order]



tcs = tcs[:num_traces,...]
so = so[:num_traces]
masks = masks[:num_traces,...]

#now sort back in position
llocs = np.array([ndimage.measurements.center_of_mass(x) for x in masks])
llocs = llocs[:,0]*masks.shape[-2] + llocs[:,1]
order2 = np.argsort(llocs)[::-1]


tcs = tcs[order2,...]
so = so[order2,...]
masks = masks[order2,...]


tcs = tcs[traces,:]
so = so[traces]
masks = masks[traces,:]

tc_filt = ndimage.gaussian_filter(tcs,(0,3))#np.array([prox_tv.tv1_1d(t,0.01) for t in tcs])



cmap = matplotlib.cm.viridis




fig = plt.figure(constrained_layout = True)
gs  = fig.add_gridspec(2,5)
ax = fig.add_subplot(gs[:,-2:])
colors = []

event_times = []
for i in range(len(traces)):
    #ax.plot([0,tcs.shape[-1]*T],np.ones(2)*i*100/sep,'k',alpha = 0.5)
    line = ax.plot(np.arange(tcs.shape[-1])*T,(tcs[i]-1)*100 + i*100/sep, color = cmap(i/len(traces)))
    _ = ax.plot(np.arange(tcs.shape[-1])*T,(tc_filt[i]-1)*100 + i*100/sep, color = 'k')
    colors.append(line[0].get_c())
    ev = events[so[i]]
    evp = event_props[so[i]]
    ax.text(-10,(i-0.15)*100/sep,f'{i}',fontdict = {'fontsize':14},color = colors[i],ha = 'right',va = 'center')
    if True:
        event_times.append([])
        for edx,l in enumerate(ev.T):
            if evp[edx][-1] < 0:
                event_times[-1].append(np.mean(l)*T)
                #ax.fill_betweenx([(i-0.5*0.9)*100/sep,(i+0.5*0.9)*100/sep],l[0]*T,l[1]*T,facecolor = 'r',alpha = 0.5)

plt.axis('off')
pf.plot_scalebar(ax, 0, (tcs[:num_traces].min()-1)*100, 200,3,thickness = 3)

colors = (np.array(colors)*255).astype(np.uint8)
#colors = np.hstack([colors,np.ones((colors.shape[0],1))])

over = masks[:len(traces)]
struct = np.zeros((3,3,3))
struct[1,...] = 1
over = np.logical_xor(ndimage.binary_dilation(over,structure = struct,iterations = 3),over).astype(int)
over = np.sum(over[...,None]*colors[:,None,None,:],0).astype(np.uint8)
length = int(im_scalebar_length_um/1.04)

over[-20:-15,-length-10:-10] = np.ones(4,dtype = np.uint8)*255

ax1 = fig.add_subplot(gs[:,:-2])
ax1.imshow(im,cmap = 'Greys_r')
ax1.imshow(over)
plt.axis('off')
pf.label_roi_centroids(ax1, masks[:num_traces,...], colors/255,fontdict = {'fontsize':8})

fig.savefig('')

binned_spikes = corrf.bin_times(event_times,10)
binned_spikes_shuffled = np.copy(binned_spikes)
[np.random.shuffle(x) for x in binned_spikes_shuffled]