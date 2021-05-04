#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:10:19 2021

@author: peter
"""
import numpy as np
import matplotlib.cm
import matplotlib

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pyqtgraph as pg

#import catch22

#import prox_tv

from pathlib import Path

import pandas as pd

from vsd_cancer.functions import cancer_functions as canf

import f.image_functions as imf

import f.plotting_functions as pf

trial_string = 'cancer_20201215_slip2_area1_long_acq_corr_long_acq_blue_0.0296_green_0.0765_heated_to_37_1'


top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/example')
if not figsave.is_dir():
    figsave.mkdir(parents = True)
    
for idx,data in enumerate(df.itertuples()):
    if data.trial_string == trial_string:
        break
    
idx = 2


trial_save = Path(save_dir,'ratio_stacks',trial_string)
    

im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
  
masks = canf.lab2masks(seg)

tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
#tcs = ndimage.gaussian_filter(tcs,(0,3))

event_dict = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

exc_circ = event_dict['excluded_circle']
events = event_dict['events'][idx]
thresh = event_dict['detection_params']['thresh_range'][idx]

keep = [x for x in np.arange(tcs.shape[0]) if x not in exc_circ]

#sort by event amounts 
sort_order = np.array([np.sum(np.abs(events['event_props'][x][:,-1])) if x in events.keys() else 0 for x in range(tcs.shape[0])])

tcs = tcs[keep,:]
masks = masks[keep,...]
sort_order = np.argsort(sort_order[keep])[::-1]

tcs = tcs[sort_order,:]
masks = masks[sort_order,:]
so = np.array(keep)[sort_order]

num = 5

tcs = tcs[:num,...]
so = so[:num]
masks = masks[:num,...]

#now sort back in position
llocs = np.array([ndimage.measurements.center_of_mass(x) for x in masks])
llocs = llocs[:,0]*masks.shape[-2] + llocs[:,1]
order2 = np.argsort(llocs)[::-1]


tcs = tcs[order2,...]
so = so[order2,...]
masks = masks[order2,...]

tc_filt = ndimage.gaussian_filter(tcs,(0,3))#np.array([prox_tv.tv1_1d(t,0.01) for t in tcs])

T = 0.2
sep = 35

cmap = matplotlib.cm.viridis

fig = plt.figure(constrained_layout = True)
gs  = fig.add_gridspec(2,5)
ax = fig.add_subplot(gs[:,-2:])
colors = []
for i in range(num):
    ax.plot([0,tcs.shape[-1]*T],np.ones(2)*i*100/sep,'k',alpha = 0.5)
    line = ax.plot(np.arange(tcs.shape[-1])*T,(tcs[i]-1)*100 + i*100/sep, color = cmap(i/num))
    line2 = ax.plot(np.arange(tcs.shape[-1])*T,(tc_filt[i]-1)*100 + i*100/sep, color = 'k')
    colors.append(line[0].get_c())
    ev = events[so[i]]
    ax.text(-60,(i-0.15)*100/sep,f'{i}',fontdict = {'fontsize':24},color = colors[i])
    if False:
        for l in ev.T:
            ax.fill_betweenx([(i-0.5*0.9)*100/sep,(i+0.5*0.9)*100/sep],l[0]*T,l[1]*T,facecolor = 'r',alpha = 0.5)

plt.axis('off')
pf.plot_scalebar(ax, 0, (tcs[:num].min()-1)*100, 200,3,thickness = 3)

colors = (np.array(colors)*255).astype(np.uint8)
#colors = np.hstack([colors,np.ones((colors.shape[0],1))])

over = masks[:num]
struct = np.zeros((3,3,3))
struct[1,...] = 1
over = np.logical_xor(ndimage.binary_dilation(over,structure = struct,iterations = 2),over).astype(int)
over = np.sum(over[...,None]*colors[:,None,None,:],0).astype(np.uint8)
length = int(100/1.04)

over[-20:-15,-length-10:-10] = np.ones(4,dtype = np.uint8)*255

ax1 = fig.add_subplot(gs[:,:-2])
ax1.imshow(im,cmap = 'Greys_r')
ax1.imshow(over)
plt.axis('off')
pf.label_roi_centroids(ax1, masks[:num,...], colors/255)

fig.savefig(Path(figsave,'example_time_courses.png'),bbox_inches = 'tight',dpi = 300)

print([x for x in so if x in events['surround_events'].keys()])


#now plot the example with surround
cell = 91
sep = 3
tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))[cell,...]
surround_tc = events['surround_events']['tc_filt'][cell,:]
eve = events[cell]
tc_test = ndimage.gaussian_filter(tc,3)

t = np.arange(tc.shape[-1])*T
tc_filt = events['tc_filt'][cell,...]
fig2,ax2 = plt.subplots()
ax2.plot(t,(tc-1)*100,'k',linewidth = 2)
#ax2.plot(t,(tc_test-1)*100,'r',linewidth = 1)

offset = -sep
ax2.plot([0,t.max()],np.array([0,0])+offset,'r',linewidth = 1,alpha = 0.7)
ax2.plot(t,(tc_filt-1)*100+offset,'k',linewidth = 2)
#ax2b.plot(t,(tc_test-1)*100,'k',linewidth = 2,alpha = 0.4)
ax2.plot([0,t.max()],np.array([-1,1])[None,:]*np.array([thresh*100,thresh*100])[:,None]+offset,'--r',linewidth = 2)
for l in eve.T:
    ax2.fill_betweenx(np.array([tc_filt.min()-1+offset/(100*1.1),tc.max()-1])*1.1*100,l[0]*T,(l[1]-1)*T,facecolor = 'r',alpha = 0.5)
    
plt.axis('off')
pf.plot_scalebar(ax2, 0, 1.1*(tc_filt.min()-1)*100+offset, 100, 0.5,thickness = 3)

fig.savefig(Path(figsave,'example_detection.png'),bbox_inches = 'tight',dpi = 300)