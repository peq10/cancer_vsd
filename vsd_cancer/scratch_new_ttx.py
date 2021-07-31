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

use = [x for x in np.unique(df['exp_stage']) if 'TTX' in x]


T = 0.2



def plot_events_TTX_washout(df,use,log = True,upper_lim = 6.6,lower_lim = 0, T = 0.2,nbins = 20,only_neg = True):
    
    
    dfn = df.copy()
    
    
    use_bool = np.array([np.any(x in use) for x in dfn.exp_stage])
    dfn = dfn[use_bool]
    
    
    too_big = np.abs(dfn.event_amplitude) > upper_lim/100
    too_small =  np.abs(dfn.event_amplitude) < lower_lim/100
    dfn = dfn[np.logical_not(np.logical_or(too_big,too_small))]
    
    if only_neg:
        
        dfn = dfn[dfn['event_amplitude'] < 0]
    
    length_bins = np.histogram(dfn['event_length']*T,bins = nbins)[1]
    amp_bins = np.histogram(np.abs(dfn['event_amplitude'])*100,bins = nbins)[1]
    
    neg = dfn[dfn.stage == 'pre']
    pos = dfn[dfn.stage == 'post']
    wash = dfn[dfn.stage == 'washout']
    
    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=0.3, hspace=0.3) 
    
    fig,axarr = plt.subplots(figsize = (8,6))
    
    histtype = 'bar'
    
    ax0 = plt.subplot(gs[0])
    ax0.hist(np.abs(neg['event_amplitude'])*100,bins = amp_bins,log = log,label = 'Pre', histtype = histtype)
    ax0.hist(np.abs(pos['event_amplitude'])*100,bins = amp_bins,log = log,label = 'TTX 10 $\mathrm{\mu}$M', histtype = histtype)
    ax0.hist(np.abs(wash['event_amplitude'])*100,bins = amp_bins,log = log,label = 'Washout', histtype = histtype)
    ax0.set_xlabel('Absolute event amplitude (% $\Delta$R/R$_0$)')
    ax0.set_ylabel('Observed Frequency')    
    ax0.legend(frameon = False)
    
    ax1 = plt.subplot(gs[1])
    ax1.hist(np.abs(neg['event_length'])*T,bins = length_bins,log = log,label = 'Pre', histtype = histtype)
    ax1.hist(np.abs(pos['event_length'])*T,bins = length_bins,log = log,label = 'TTX 10 $\mathrm{\mu}$M', histtype = histtype)
    ax1.hist(np.abs(pos['event_length'])*T,bins = length_bins,log = log,label = 'Washout', histtype = histtype)
    ax1.set_xlabel('Event length (s)')
    ax1.set_ylabel('Observed Frequency')   
    ax1.legend(frameon = False)
    
    
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    
    ax2 = plt.subplot(gs[3])
    ax2.hist2d(np.abs(neg['event_amplitude'])*100,neg['event_length']*T,bins = (amp_bins,length_bins),norm = norm)
    ax2.set_xlabel('Pre-TTX event amplitude (% $\Delta$R/R$_0$)')    
    ax2.set_ylabel('Event length (s)')
    
    ax3 = plt.subplot(gs[4])
    #ax3.sharey(ax2)
    #ax3.sharex(ax2)
    print('clipping x?')
    ax3.hist2d(np.abs(pos['event_amplitude'])*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    ax3.set_xlabel('Post-TTX event size (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event length (s)')
    
    ax3 = plt.subplot(gs[5])
    #ax3.sharey(ax2)
    #ax3.sharex(ax2)
    print('clipping x?')
    ax3.hist2d(np.abs(wash['event_amplitude'])*100,wash['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    ax3.set_xlabel('Washout event size (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event length (s)')
    
    return fig

    

fig1 = plot_events_TTX_washout(df,use,log = True)

fig2 = plot_events_TTX_washout(df,use,log = False)
