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
    figsave = Path(figure_dir,'10A_figure')
    if not figsave.is_dir():
        figsave.mkdir()
    
    plot_MCF_hist(save_dir,figsave,filetype)
    pass

def plot_MCF_hist(save_dir,figsave,filetype):
    
    df = pd.read_csv(Path(save_dir,'all_events_df.csv'))
    df['exp_stage'] = df.expt + '_' + df.stage

    use = ['MCF10A_TGFB_none', 'MCF10A_none']
    
    

    log = [True,False]
    only_neg = [True,False]
    histtype = ['bar','step']

    for l in log:
        for n in only_neg:
            for h in histtype:
                fig = plot_events_MCF(df,use,log = l,only_neg=n,histtype = h)
                fig.savefig(Path(figsave,'histograms',f'MCF_histograms_{h}_log_{l}_onlyneg_{n}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)


def plot_events_MCF(df,use,log = True,upper_lim = 6.6,lower_lim = 0, T = 0.2,nbins = 20,only_neg = True,histtype = 'step'):
    
    
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
    
    neg = dfn[dfn.exp_stage == 'MCF10A_none']
    pos = dfn[dfn.exp_stage == 'MCF10A_TGFB_none']
    
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.3, hspace=0.3) 
    
    fig,axarr = plt.subplots(figsize = (8,6))
    
    ax0 = plt.subplot(gs[0])
    ax0.hist(np.abs(neg['event_amplitude'])*100,bins = amp_bins,log = log,label = 'MCF10A',histtype = histtype)
    ax0.hist(np.abs(pos['event_amplitude'])*100,bins = amp_bins,log = log,label = 'MCF10A + TGF-$\\beta$',histtype = histtype)
    ax0.set_xlabel('Absolute event amplitude (% $\Delta$R/R$_0$)')
    ax0.set_ylabel('Observed Frequency')    
    ax0.legend(frameon = False)
    
    ax1 = plt.subplot(gs[1])
    ax1.hist(np.abs(neg['event_length'])*T,bins = length_bins,log = log,label = 'MCF10A',histtype = histtype)
    ax1.hist(np.abs(pos['event_length'])*T,bins = length_bins,log = log,label = 'MCF10A + TGF-$\\beta$',histtype = histtype)
    ax1.set_xlabel('Event length (s)')
    ax1.set_ylabel('Observed Frequency')   
    ax1.legend(frameon = False)
    
    
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    
    ax2 = plt.subplot(gs[2])
    ax2.hist2d(np.abs(neg['event_amplitude'])*100,neg['event_length']*T,bins = (amp_bins,length_bins),norm = norm)
    ax2.set_xlabel('MCF10A event amplitude (% $\Delta$R/R$_0$)')    
    ax2.set_ylabel('Event length (s)')
    
    ax3 = plt.subplot(gs[3])
    ax3.hist2d(np.abs(pos['event_amplitude'])*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    ax3.set_xlabel('MCF10A + TGF-$\\beta$ event size (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event length (s)')
    
    return fig
if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    make_figures(initial_df,save_dir,figure_dir)