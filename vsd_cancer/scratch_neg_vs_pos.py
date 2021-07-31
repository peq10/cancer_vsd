#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 09:59:17 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.gridspec as gridspec

import pandas as pd

from pathlib import Path

import f.plotting_functions as pf


top_dir = Path('/home/peter/data/Firefly/cancer')

data_dir = Path(top_dir,'analysis','full')

df = pd.read_csv(Path(data_dir,'all_events_df.csv'))


#df = df[df.expt == 'standard']


df['exp_stage'] = df.expt + '_' + df.stage

use = ['TTX_10um_washout_pre','TTX_1um_pre','TTX_10um_pre','standard_none']


T = 0.2



def plot_events(df,use,log = True,upper_lim = 6.6,lower_lim = 0, T = 0.2):
    
    
    dfn = df.copy()
    
    
    use_bool = np.array([np.any(x in use) for x in dfn.exp_stage])
    dfn = dfn[use_bool]
    
    
    too_big = np.abs(dfn.event_amplitude) > 6.6/100
    too_small =  np.abs(dfn.event_amplitude) < 0/100
    dfn = dfn[np.logical_not(np.logical_or(too_big,too_small))]
    
    
    neg = dfn[dfn.event_amplitude <0]
    pos = dfn[dfn.event_amplitude >0]
    
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.3, hspace=0.3) 
    
    fig,axarr = plt.subplots(figsize = (8,6))
    
    ax0 = plt.subplot(gs[0])
    ax0.hist(np.abs(neg['event_amplitude'])*100,bins = 50,log = log,label = '-ve')
    ax0.hist(pos['event_amplitude']*100,bins = 50,log = log,label = '+ve')
    ax0.set_xlabel('Absolute event amplitude (% $\Delta$F/F$_0$')
    ax0.set_ylabel('Observed Frequency')    
    ax0.legend(frameon = False)
    
    ax1 = plt.subplot(gs[1])
    ax1.hist(np.abs(neg['event_length'])*T,bins = 50,log = log,label = '-ve')
    ax1.hist(pos['event_length']*T,bins = 50,log = log,label = '+ve')
    ax1.set_xlabel('Event length (s)')
    ax1.set_ylabel('Observed Frequency')   
    ax1.legend(frameon = False)
    
    
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    
    ax2 = plt.subplot(gs[2])
    ax2.hist2d(np.abs(neg['event_amplitude'])*100,neg['event_length'],bins = 50,norm = norm)
    ax2.set_xlabel('Negative event amplitude (% $\Delta$F/F$_0$)')    
    ax2.set_ylabel('Event length (s)')
    
    ax3 = plt.subplot(gs[3])
    ax3.sharey(ax2)
    ax3.sharex(ax2)
    print('clipping x?')
    ax3.hist2d(np.abs(pos['event_amplitude'])*100,pos['event_length'],bins = 50,norm=norm)
    ax3.set_xlabel('Positive event size (% $\Delta$F/F$_0$)')    
    ax3.set_ylabel('Event length (s)')
    
    return fig

    

fig1 = plot_events(df,use,log = True)

fig2 = plot_events(df,use,log = False)
