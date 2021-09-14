#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:32:43 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.gridspec as gridspec

import pandas as pd

from pathlib import Path

import f.plotting_functions as pf

import scipy.stats as stats

import astropy.stats as ass

import scipy.ndimage as ndimage
from vsd_cancer.functions import stats_functions as statsf
from vsd_cancer.functions import cancer_functions as canf
import matplotlib.cm

def get_most_active_traces(num_traces,df,trial_save,trial_string):
    tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    event_dict = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

    idx = 1
    events = event_dict['events'][idx]
    
    keep = [x for x in np.arange(tcs.shape[0])]
    
    #sort by event amounts 
    sort_order = np.array([np.sum(np.abs(events['event_props'][x][:,-1])) if x in events.keys() else 0 for x in range(tcs.shape[0])])
    
    t = np.copy(sort_order)
    t.sort()
    print(t[-num_traces:])
    
    tcs = tcs[keep,:]
    sort_order = np.argsort(sort_order[keep])[::-1]
    
    tcs = tcs[sort_order,:]
    so = np.array(keep)[sort_order]
    
    
    
    tcs = ndimage.gaussian_filter(tcs[:num_traces,...],(0,3))
    so = so[:num_traces]
    
    return tcs,so
    
top_dir = Path('/home/peter/data/Firefly/cancer')

data_dir = Path(top_dir,'analysis','full')

df = pd.read_csv(Path(data_dir,'non_ttx_active_df_by_cell.csv'))


initial_df = pd.read_csv(Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv'))




mc_str = 'cancer_20210119_slip2_area1_long_acq_corr_long_acq_blue_0.0454_green_0.0671_1'
tg_str = 'cancer_20210122_slip1_area3_long_acq_MCF10A_tgfbeta_long_acq_blue_0.02909_green_0.0672_1'

tds = [mc_str,tg_str]

at = []
att = []
T = 0.2
sep = 50
num_traces = 5
im_scalebar_length_um = 100
cells = [[185,105,109,191,149],[19,78, 51, 45, 71]]
#cells = [get_most_active_traces(5,initial_df,Path(data_dir,'ratio_stacks',mc_str),mc_str)[1],get_most_active_traces(5,initial_df,Path(data_dir,'ratio_stacks',tg_str),tg_str)[1]]

for idx,t in enumerate(tds):
    trial_save = Path(data_dir,'ratio_stacks',t)
    
    im = np.load(Path(trial_save,f'{t}_im.npy'))
    
    seg = np.load(Path(trial_save,f'{t}_seg.npy'))
    mask = canf.lab2masks(seg)
    
    
    tcs = np.load(Path(trial_save,f'{t}_all_tcs.npy'))
    att.append(tcs)
    
    at.append(ndimage.gaussian_filter(tcs[cells[idx]],(0,3)))
    
    
    tc_filt = ndimage.gaussian_filter(tcs[cells[idx]],(0,3)) 
    tcs = tcs[cells[idx]]

    masks = mask[cells[idx]]



    cmap = matplotlib.cm.tab10    
    

    fig = plt.figure(constrained_layout = True)
    gs  = fig.add_gridspec(2,5)
    ax = fig.add_subplot(gs[:,-2:])
    colors = []
    for i in range(num_traces):
        ax.plot([0,tcs.shape[-1]*T],np.ones(2)*i*100/sep,'k',alpha = 0.5)
        line = ax.plot(np.arange(tcs.shape[-1])*T,(tcs[i]-1)*100 + i*100/sep, color = cmap(i/num_traces))
        _ = ax.plot(np.arange(tcs.shape[-1])*T,(tc_filt[i]-1)*100 + i*100/sep, color = 'k')
        colors.append(line[0].get_c())

        ax.text(-10,(i-0.15)*100/sep,f'{i}',fontdict = {'fontsize':14},color = colors[i],ha = 'right',va = 'center')

    plt.axis('off')
    pf.plot_scalebar(ax, 0, (tcs[:num_traces].min()-1)*100, 200,1,thickness = 3)
    
    colors = (np.array(colors)*255).astype(np.uint8)
    #colors = np.hstack([colors,np.ones((colors.shape[0],1))])
    
    over = masks[:num_traces]
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


        
