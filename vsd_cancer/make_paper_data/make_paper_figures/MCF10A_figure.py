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
import datetime

import f.plotting_functions as pf


from vsd_cancer.functions import stats_functions as statsf

def make_figures(initial_df,save_dir,figure_dir,filetype = '.png', redo_stats = False):
    figsave = Path(figure_dir,'10A_figure')
    if not figsave.is_dir():
        figsave.mkdir()
    
    plot_MCF_hist(save_dir,figsave,filetype)

    plot_compare_mda(save_dir,figsave,filetype , 'neg_event_rate', np.mean, 'np.mean', scale = 3, density = False, redo_stats =redo_stats)
    plot_compare_mda(save_dir,figsave,filetype , 'neg_event_rate', np.mean, 'np.mean', scale = 3, density = True, redo_stats = False)
    plot_compare_mda(save_dir,figsave,filetype , 'neg_integ_rate', np.mean, 'np.mean', scale = 3, density = False, redo_stats = redo_stats)
    plot_compare_mda(save_dir,figsave,filetype , 'neg_integ_rate', np.mean, 'np.mean', scale = 3, density = True, redo_stats = False)


def plot_compare_mda(save_dir,figsave,filetype , key, function, function_name, scale = 3, density = False, redo_stats = True,num_resamplings = 10**6):
    df = pd.read_csv(Path(save_dir,'non_ttx_active_df_by_cell.csv'))

    T = 0.2
    
    
    df['exp_stage'] = df.expt + '_' + df.stage
    df['day_slip'] = df.day.astype(str) + '_' + df.slip.astype(str) 
    
    
    df['event_rate'] = (df['n_neg_events'] +  df['n_pos_events'])/(df['obs_length']*T)
    df['neg_event_rate'] = (df['n_neg_events'] )/(df['obs_length']*T)
    
    df['integ_rate'] = (df['integrated_events'])/(df['obs_length']*T)
    df['neg_integ_rate'] = -1*(df['neg_integrated_events'] )/(df['obs_length']*T)
    
    mda = df[df.exp_stage == 'standard_none'][[key,'day_slip']]
    mcf = df[df.exp_stage == 'MCF10A_none'][[key,'day_slip']]
    tgf = df[df.exp_stage == 'MCF10A_TGFB_none'][[key,'day_slip']]
    
    md = mda[key].to_numpy()
    mc = mcf[key].to_numpy()
    tg = tgf[key].to_numpy()
    
    
    bins = np.histogram(np.concatenate((md,mc,tg))*10**3,bins = 20)[1]
    

    fig,axarr = plt.subplots(nrows = 3)
    c = 0.05
    axarr[0].hist(md*10**scale,bins = bins, log = True, density = density, label = 'MDA-MB-231', color = (c,c,c))
    axarr[1].hist(mc*10**scale,bins = bins, log = True,  density = density, label = 'MCF10A', color = (c,c,c))
    axarr[2].hist(tg*10**scale,bins = bins, log = True,  density = density, label = 'MCF10A+TGF$\\beta$', color = (c,c,c))
    
    for idx,a in enumerate(axarr):
        if not density:
            a.set_ylim([0.6,10**4.5])
            a.set_yticks(10**np.arange(0,4,3))
        a.legend(frameon = False,loc = (0.4,0.4),fontsize = 16)
        pf.set_all_fontsize(a, 16)
        if idx != 2:
            a.set_xticklabels([])
            
        
    if not density:
        axarr[1].set_ylabel('Number of cells')
    else:
        axarr[1].set_ylabel('Proportion of cells')

    if key == 'neg_event_rate':
        axarr[-1].set_xlabel('Negative event rate ' + '(1000 cells$^{-1}$ s$^{-1}$)')
    elif key == 'neg_integ_rate':
        axarr[-1].set_xlabel(f'Integrated event rate per {10**scale} cells ' + '(%$\cdot$s / s)')
    else:
        raise ValueError('wrong key')
        
    fig.savefig(Path(figsave,f'MCF_compare_density_{density}_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

    if redo_stats:
        p_mda_mcf,_,f1 = statsf.bootstrap_test(md,mc,function = function,plot = True,num_resamplings = num_resamplings, names = ['MDA-MB-231', 'MCF10A'])
        p_mda_tgf,_,f2 = statsf.bootstrap_test(md,tg,function = function,plot = True,num_resamplings = num_resamplings, names = ['MDA-MB-231', 'MCF10A + TGF$\\beta$'])
        p_mcf_tgf,_,f3 = statsf.bootstrap_test(tg,mc,function = function,plot = True,num_resamplings = num_resamplings, names = ['MCF10A + TGF$\\beta$', 'MCF10A'])
        
        f1.savefig(Path(figsave,'bootstrap',f'bootstrap_231_MCF_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        f2.savefig(Path(figsave,'bootstrap',f'bootstrap_231_tgf_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        f3.savefig(Path(figsave,'bootstrap',f'bootstrap_tgf_MCF_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        
        
        with open(Path(figsave, f'statistical_test_results_{key}.txt'),'w') as f:
            f.write(f'{datetime.datetime.now()}\n')
            f.write(f'Testing significance of second less than first for function {function_name}\n')
            f.write(f'N cells MDA: {len(md)}\n')
            f.write(f'N cells MCF: {len(mc)}\n')
            f.write(f'N cells TGF: {len(tg)}\n')
            f.write(f'N slips MDA: {len(np.unique(mda["day_slip"]))}\n')
            f.write(f'N slips MCF: {len(np.unique(mcf["day_slip"]))}\n')
            f.write(f'N slips TGF: {len(np.unique(tgf["day_slip"]))}\n')
    
            f.write(f'Num resamples: {num_resamplings}\n')
            f.write(f'p MDA-MCF {p_mda_mcf}\n')
            f.write(f'p MDA-TGF {p_mda_tgf}\n')
            f.write(f'p MCF-TGF {p_mcf_tgf}\n')


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
    amp_bins = np.histogram(dfn['event_amplitude']*100,bins = nbins)[1]
    
    neg = dfn[dfn.exp_stage == 'MCF10A_none']
    pos = dfn[dfn.exp_stage == 'MCF10A_TGFB_none']
    
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.3, hspace=0.3) 
    
    fig,axarr = plt.subplots(figsize = (8,6))
    
    ax0 = plt.subplot(gs[0])
    ax0.hist(neg['event_amplitude']*100,bins = amp_bins,log = log,label = 'MCF10A',histtype = histtype)
    ax0.hist(pos['event_amplitude']*100,bins = amp_bins,log = log,label = 'MCF10A + TGF-$\\beta$',histtype = histtype)
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
    ax2.hist2d(neg['event_amplitude']*100,neg['event_length']*T,bins = (amp_bins,length_bins),norm = norm)
    ax2.set_xlabel('MCF10A event amplitude (% $\Delta$R/R$_0$)')    
    ax2.set_ylabel('Event length (s)')
    
    ax3 = plt.subplot(gs[3])
    ax3.hist2d(pos['event_amplitude']*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    ax3.set_xlabel('MCF10A + TGF-$\\beta$ event size (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event length (s)')
    
    return fig
if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    make_figures(initial_df,save_dir,figure_dir)