#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:44:41 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import matplotlib.cm

import scipy.ndimage as ndimage

import scipy.stats as stats

import astropy.visualization as av
import astropy.stats as ass

from vsd_cancer.functions import cancer_functions as canf

from pathlib import Path

import f.plotting_functions as pf
import f.general_functions as gf
import pyqtgraph as pg
import tifffile


top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/waves')
if not figsave.is_dir():
    figsave.mkdir(parents = True)

expts_use = ['standard','TTX_10um','TTX_10_um_washout','TTX_1um']
use = [x in expts_use for x in df.expt]

stage_use = ['nan','pre']
use2 = [str(x) in stage_use for x in df.stage]

df = df[(df.use == 'y') & (use) & (use2)]

for idx,data in enumerate(df.itertuples()):
    if '20201216_slip1_area2_long_acq' in data.trial_string:
        break
    
trial_string = data.trial_string
trial_save = Path(save_dir,'ratio_stacks',trial_string)



im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
seg =  np.load(Path(trial_save,f'{trial_string}_seg.npy'))



masks = canf.lab2masks(seg)

T = 0.2





tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))[:,-1000:-400]
tc_filt=ndimage.gaussian_filter(tcs,(0,3))

loc = 300,550

ac = tc_filt[:,loc[0]:loc[1]]  < 0.99

roi_ids = np.where(np.sum(ac,-1) >0)[0]
wh = np.where(ac)
start_ids = wh[1][np.unique(wh[0],return_index = True)[1]]

sort = np.argsort(start_ids)
roi_ids = roi_ids[sort]
start_ids = start_ids[sort]

roi_t = tcs[roi_ids,:]
roi_t_filt = ndimage.gaussian_filter(roi_t,(0,1))

#now plot the dist/time stuff
roi_masks = masks[roi_ids,...]
coms = np.array([ndimage.center_of_mass(mask) for mask in roi_masks])

dists = np.sqrt((coms[:,0] - coms[0,0])**2 + (coms[:,1] - coms[0,1])**2)*1.04
times = np.argmin(roi_t_filt,axis = -1)*T
times -= times[0]

sizes = np.sum(np.abs(roi_t[:,loc[0]:loc[1]]),-1)
sizes = np.min(roi_t[:,loc[0]:loc[1]],-1)
sizes -= np.mean(roi_t[:,loc[0]-20:loc[0]])
sizes *= 100

fit = stats.linregress(times,dists)

fig,ax = plt.subplots()
ax.plot(times,fit.slope*times +fit.intercept,'k',linewidth = 2)
ax.plot(times,dists,'.r',markersize = 10,label = 'Cell event peaks')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance ($\mathrm{\mu}$m)')
ax.set_xticks([0,5,10,15])
ax.set_yticks([0,250,500])
ax.text(0,400,f'Fit speed: {fit.slope:.0f}'+' $\mathrm{\mu}$m/s\nr$^2$ ='+f' {fit.rvalue**2:.2f}')
plt.legend(frameon = False, loc = (0,0.7))
pf.set_thickaxes(ax, 3)
pf.make_square_plot(ax)
pf.set_all_fontsize(ax, 14)
fig.savefig(Path(figsave,'speed_fit.png'))

#plot distance/magnitude
fit2 = stats.linregress(dists,sizes)
fig,ax = plt.subplots()
ax.plot(dists,sizes,'.r',markersize = 10,label = 'Cell peak amplitudes')
ax.set_yticks([-6,-3,0])
ax.set_xticks([0,250,500])
ax.set_ylabel('Transient peak amplitude (%)')
ax.set_xlabel('Distance ($\mathrm{\mu}$m)')
pf.set_thickaxes(ax, 3)
pf.make_square_plot(ax)
pf.set_all_fontsize(ax, 14)
fig.savefig(Path(figsave,'speed_fit.png'))


sep = 20#
norm_dist = dists/dists.max()
num = roi_ids.shape[0]
cmap = matplotlib.cm.viridis

fig = plt.figure(constrained_layout = True)
gs  = fig.add_gridspec(2,5)
ax = fig.add_subplot(gs[:,-2:])
colors = []
for i in range(num):
    line = ax.plot(np.arange(roi_t.shape[-1])*T,(roi_t[i]-1)*100 + i*100/sep, color = cmap(i/num))
    line2 = ax.plot(np.arange(roi_t.shape[-1])*T,(roi_t_filt[i]-1)*100 + i*100/sep, color = 'k')
    colors.append(line[0].get_c())
    ax.text(-60,(i-0.15)*100/sep,f'{i}',fontdict = {'fontsize':24},color = colors[i])


plt.axis('off')
pf.plot_scalebar(ax, 0, (tcs[:num].min()-1)*100 - 3, 20,3,thickness = 3)

colors = (np.array(colors)*255).astype(np.uint8)
#colors = np.hstack([colors,np.ones((colors.shape[0],1))])

over = roi_masks
struct = np.zeros((3,3,3))
struct[1,...] = 1
over = np.logical_xor(ndimage.binary_dilation(over,structure = struct,iterations = 2),over).astype(int)
over = np.sum(over[...,None]*colors[:,None,None,:],0).astype(np.uint8)
length = int(100/1.04)

over[-20:-15,10:10+length] = np.ones(4,dtype = np.uint8)*255

ax1 = fig.add_subplot(gs[:,:-2])
ax1.imshow(im,cmap = 'Greys_r')
ax1.imshow(over)
plt.axis('off')
pf.label_roi_centroids(ax1, roi_masks, colors/255)

fig.savefig(Path(figsave,'wave_time_courses.png'),bbox_inches = 'tight',dpi = 300)

ex_roi = 185
ex_t = tcs[ex_roi]

'''

rat = np.load(Path(trial_save,f'{trial_string}_ratio_stack.npy'))[-1000:-400]
rat2 =ndimage.gaussian_filter(rat,(3,1,1))
#color balance
cmin = np.percentile(rat2,0.05)
cmax = np.percentile(rat2,99.95)
rat2[np.where(rat2<cmin)] = cmin
rat2[np.where(rat2>cmax)] = cmax
rat2 = gf.to_8_bit(rat2)
rat2 = rat2[...,None]*np.ones(3,dtype = np.uint8)[None,None,None,:]
over = np.ones(rat2.shape)*over[None,...,:-1]
wh = np.where(over)
rat2[wh] = over[wh]


stack = tifffile.imread(data.tif_file)
stack = stack[1::2]
stack = stack[-1000:-400]
#balance the colors for easier visualisation
cmin = np.percentile(stack,0.05)
cmax = np.percentile(stack,99.95)
stack[np.where(stack<cmin)] = cmin
stack[np.where(stack>cmax)] = cmax
stack = gf.to_8_bit(stack)
stack = stack[...,None]*np.ones(3,dtype = np.uint8)[None,None,None,:]
stack[wh] = over[wh]

tifffile.imsave(Path(figsave,'wave_vid.tif'),rat2)
tifffile.imsave(Path(figsave,'wave_vid_im.tif'),stack)
tifffile.imsave(Path(figsave,'im.tif'),gf.to_16_bit(im))

'''