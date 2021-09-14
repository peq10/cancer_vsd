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

mc = df[df.expt == 'MCF10A']
tg = df[df.expt == 'MCF10A_TGFB']

tg =tg.sort_values(by = 'n_neg_events')

mc = mc.sort_values(by = 'n_neg_events')


mc_str = mc.iloc[-2].trial#'cancer_20210313_slip4_area1_long_acq_corr_MCF10A_37deg_long_acq_blue_0.039_green_0.04734_1'
tg_str = tg.iloc[-1].trial#'cancer_20210122_slip1_area3_long_acq_MCF10A_tgfbeta_long_acq_blue_0.02909_green_0.0672_1'

tds = [mc_str,tg_str]

at = []
att = []

cells = [[185,105,109,191,149],[19,78, 51, 45, 71]]
cells = [get_most_active_traces(5,initial_df,Path(data_dir,'ratio_stacks',mc_str),mc_str)[1],get_most_active_traces(5,initial_df,Path(data_dir,'ratio_stacks',tg_str),tg_str)[1]]

for idx,t in enumerate(tds):
    trial_save = Path(data_dir,'ratio_stacks',t)
    
    im = np.load(Path(trial_save,f'{t}_im.npy'))
    
    seg = np.load(Path(trial_save,f'{t}_seg.npy'))
    mask = canf.lab2masks(seg)
    
    
    tcs = np.load(Path(trial_save,f'{t}_all_tcs.npy'))
    att.append(tcs)
    
    at.append(ndimage.gaussian_filter(tcs[cells[idx]],(0,3)))
    
    
    ma = mask[cells[idx]]
    overlay,colours = pf.make_colormap_rois(ma,matplotlib.cm.viridis)

    plt.imshow(im, cmap = 'Greys_r')
    plt.imshow(overlay)
    plt.show()
    plt.plot(at[-1].T + np.arange(5)/100)
    plt.show()
        
        #rat = np.load(Path(trial_save,f'{t}_ratio_stack.npy'))