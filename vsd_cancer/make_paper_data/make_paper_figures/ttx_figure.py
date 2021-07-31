#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:33:36 2021

@author: peter
"""
from pathlib import Path


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


import matplotlib.cm
import matplotlib.gridspec as gridspec
import matplotlib as mpl

def make_figures(initial_df,save_dir,figure_dir,filetype = '.png'):
    figsave = Path(figure_dir,'ttx_figure')
    if not figsave.is_dir():
        figsave.mkdir()
    
    plot_TTX_pre_post(save_dir,figsave,filetype)
    plot_TTX_washout(save_dir,figsave,filetype)
    pass


def plot_TTX_pre_post(save_dir,figsave,filetype):
    
    df = pd.read_csv(Path(save_dir,'all_events_df.csv'))
    df['exp_stage'] = df.expt + '_' + df.stage

    use = [x for x in np.unique(df['exp_stage']) if 'TTX' in x and 'washout_washout' not in x]
    
    
    ttx = [1,10]
    log = [True,False]
    only_neg = [True,False]
    histtype = ['bar','step']
    for t in ttx:
        for l in log:
            for n in only_neg:
                for h in histtype:
                    fig = plot_events_TTX(df,use,TTX_level = t,log = l,only_neg=n,histtype = h)
                    fig.savefig(Path(figsave,'pre_post',str(t),f'TTX_{t}um_histograms_{h}_log_{l}_onlyneg_{n}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

def plot_TTX_washout(save_dir,figsave,filetype):
    
    df = pd.read_csv(Path(save_dir,'all_events_df.csv'))
    df['exp_stage'] = df.expt + '_' + df.stage

    use = [x for x in np.unique(df['exp_stage']) if 'TTX' in x]
    
    log = [True,False]
    only_neg = [True,False]
    histtype = ['bar','step']
    
    for l in log:
        for n in only_neg:
            for h in histtype:
                fig = plot_events_TTX_washout(df,use,log = l,only_neg=n,histtype = h)
                fig.savefig(Path(figsave,'washout',f'TTX_washout_histograms_{h}_log_{l}_onlyneg_{n}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
    
def plot_events_TTX(df,use,TTX_level = 1,log = True,upper_lim = 6.6,lower_lim = 0, T = 0.2,nbins = 20,only_neg = True,histtype = 'bar'):
    
    
    dfn = df.copy()
    
    use = [x for x in use if f'{TTX_level}um' in x]
    
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
    
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.3, hspace=0.3) 
    
    fig,axarr = plt.subplots(figsize = (8,6))
    
    ax0 = plt.subplot(gs[0])
    ax0.hist(np.abs(neg['event_amplitude'])*100,bins = amp_bins,log = log,label = 'Pre',histtype = histtype)
    ax0.hist(np.abs(pos['event_amplitude'])*100,bins = amp_bins,log = log,label = f'TTX {TTX_level}'+' $\mathrm{\mu}$M',histtype = histtype)
    ax0.set_xlabel('Absolute event amplitude (% $\Delta$R/R$_0$)')
    ax0.set_ylabel('Observed Frequency')    
    ax0.legend(frameon = False)
    
    ax1 = plt.subplot(gs[1])
    ax1.hist(np.abs(neg['event_length'])*T,bins = length_bins,log = log,label = 'Pre',histtype = histtype)
    ax1.hist(np.abs(pos['event_length'])*T,bins = length_bins,log = log,label = f'TTX {TTX_level}'+' $\mathrm{\mu}$M',histtype = histtype)
    ax1.set_xlabel('Event length (s)')
    ax1.set_ylabel('Observed Frequency')   
    ax1.legend(frameon = False)
    
    
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    
    ax2 = plt.subplot(gs[2])
    ax2.hist2d(np.abs(neg['event_amplitude'])*100,neg['event_length']*T,bins = (amp_bins,length_bins),norm = norm)
    ax2.set_xlabel('Pre-TTX event amplitude (% $\Delta$R/R$_0$)')    
    ax2.set_ylabel('Event length (s)')
    
    ax3 = plt.subplot(gs[3])
    ax3.hist2d(np.abs(pos['event_amplitude'])*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    ax3.set_xlabel('Post-TTX event size (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event length (s)')
    
    return fig

def plot_events_TTX_washout(df,use,log = True,upper_lim = 6.6,lower_lim = 0, T = 0.2,nbins = 20,only_neg = True,histtype = 'bar'):
    
    
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
    ax1.hist(np.abs(wash['event_length'])*T,bins = length_bins,log = log,label = 'Washout', histtype = histtype)
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
    ax3.hist2d(np.abs(pos['event_amplitude'])*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    ax3.set_xlabel('Post-TTX event size (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event length (s)')
    
    ax3 = plt.subplot(gs[5])
    ax3.hist2d(np.abs(wash['event_amplitude'])*100,wash['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    ax3.set_xlabel('Washout event size (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event length (s)')
    
    return fig

if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    make_figures(initial_df,save_dir,figure_dir)