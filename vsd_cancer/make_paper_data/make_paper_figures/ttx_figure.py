#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:33:36 2021

@author: peter
"""
from pathlib import Path

import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from vsd_cancer.functions import stats_functions as statsf

import f.plotting_functions as pf

import matplotlib.cm
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import scipy.ndimage as ndimage

def make_figures(initial_df,save_dir,figure_dir,filetype = '.png', redo_stats = False):
    figsave = Path(figure_dir,'ttx_figure')
    if not figsave.is_dir():
        figsave.mkdir()
    
    plot_TTX_pre_post(save_dir,figsave,filetype,redo_stats)
    plot_TTX_washout(save_dir,figsave,filetype, redo_stats)

    plot_pre_post_ttx_traces(initial_df, save_dir, figsave, filetype)

def plot_pre_post_ttx_traces(initial_df, save_dir, figsave, filetype):
    def get_most_active_traces(num_traces,df,trial_save,trial_string):
        tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
        event_dict = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
    
        idx = 0
        events = event_dict['events'][idx]
        
        keep = [x for x in np.arange(tcs.shape[0])]
        
        #sort by event amounts 
        sort_order = np.array([np.sum(np.abs(events['event_props'][x][:,-1])) if x in events.keys() else 0 for x in range(tcs.shape[0])])
        
        tcs = tcs[keep,:]
        sort_order = np.argsort(sort_order[keep])[::-1]
        
        tcs = tcs[sort_order,:]
        so = np.array(keep)[sort_order]
        
        
        
        tcs = ndimage.gaussian_filter(tcs[:num_traces,...],(0,3))
        so = so[:num_traces]
        
        return tcs,so
    
    
    df = pd.read_csv(initial_df)
    ncells = 10
    T = 0.2
    
    trial_strings = ['cancer_20201216_slip1_area2_long_acq_long_acq_blue_0.0296_green_0.0765_heated_to_37_1',
                     'cancer_20201216_slip1_area3_long_acq_long_acq_blue_0.0296_green_0.0765_heated_to_37_with_TTX_1']
    tcs = []
    for t in trial_strings:
        
        print(df[df.trial_string == t].stage)
        tcs.append(get_most_active_traces(ncells,df,Path(save_dir,'ratio_stacks',t), t)[0])
        
        
    fig,ax = plt.subplots(ncols = 2)
    ax[0].plot(np.arange(tcs[0].shape[1])*T, tcs[0].T + np.arange(ncells)/20, 'k')
    ax[1].sharey(ax[0])
    ax[1].plot(np.arange(tcs[1].shape[1])*T, tcs[1].T + np.arange(ncells)/20, 'k')
    
    pf.plot_scalebar(ax[0], 0, 0.95, 100, 0.02)
    ax[0].axis('off')
    ax[1].axis('off')
        
    fig.savefig(Path(figsave,'example_traces',f'example_traces{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
    

def plot_TTX_pre_post(save_dir,figsave,filetype, redo_stats):
    
    df = pd.read_csv(Path(save_dir,'all_events_df.csv'))
    df['exp_stage'] = df.expt + '_' + df.stage

    use = [x for x in np.unique(df['exp_stage']) if 'TTX' in x and 'washout_washout' not in x]
    
    
    ttx = [1,10]
    log = [True,False]
    only_neg = [True,False]
    histtype = ['bar','step']
    
    
    ttx = [1,10]
    log = [True]
    only_neg = [False]
    histtype = ['bar']
    
    for t in ttx:
        for l in log:
            for n in only_neg:
                for h in histtype:
                    fig = plot_events_TTX(df,use,TTX_level = t,log = l,only_neg=n,histtype = h)
                    fig.savefig(Path(figsave,'pre_post',str(t),f'TTX_{t}um_histograms_{h}_log_{l}_onlyneg_{n}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

    df2 = pd.read_csv(Path(save_dir,'TTX_active_df_by_cell.csv'))
    T = 0.2
    df2['exp_stage'] = df2.expt + '_' + df2.stage
    df2['day_slip'] = df2.day.astype(str) + '_' + df2.slip.astype(str) 

    df2['neg_event_rate'] = (df2['n_neg_events'] )/(df2['obs_length']*T)
    

    df2['neg_integ_rate'] = -1*(df2['neg_integrated_events'] )/(df2['obs_length']*T)
    
    
    use2 = [x for x in np.unique(df2['exp_stage']) if 'washout' not in x]
    plot_TTX_summary(df2,use2,figsave,filetype,redo_stats = redo_stats,key = 'neg_event_rate', function = np.mean,function_name = 'np.mean',scale = 3, density = True)
    plot_TTX_summary(df2,use2,figsave,filetype,redo_stats = False,key = 'neg_event_rate', function = np.mean,function_name = 'np.mean',scale = 3, density = False)
    #plot_TTX_summary(df2,use2,figsave,filetype,redo_stats = redo_stats,key = 'neg_integ_rate', function = np.mean,function_name = 'np.mean',scale = 3, density = True)
    #plot_TTX_summary(df2,use2,figsave,filetype,redo_stats = False,key = 'neg_integ_rate', function = np.mean,function_name = 'np.mean',scale = 3, density = False)
    



def plot_TTX_washout(save_dir,figsave,filetype, redo_stats):
    
    df = pd.read_csv(Path(save_dir,'all_events_df.csv'))
    df['exp_stage'] = df.expt + '_' + df.stage

    use = [x for x in np.unique(df['exp_stage']) if 'TTX' in x and 'washout' in x]
    
    log = [True,False]
    only_neg = [True,False]
    histtype = ['bar','step']

    log = [True]
    only_neg = [False]
    histtype = ['bar']    


    for l in log:
        for n in only_neg:
            for h in histtype:

                fig = plot_events_TTX_washout(df,use,log = l,only_neg=n,histtype = h)
                fig.savefig(Path(figsave,'washout',f'TTX_washout_histograms_{h}_log_{l}_onlyneg_{n}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
    
    #now plot the mean and bootstrapped cis   
    df2 = pd.read_csv(Path(save_dir,'TTX_active_df_by_cell.csv'))
    T = 0.2
    df2['exp_stage'] = df2.expt + '_' + df2.stage
    df2['neg_event_rate'] = (df2['n_neg_events'] )/(df2['obs_length']*T)
    df2['day_slip'] = df2.day.astype(str) + '_' + df2.slip.astype(str) 
    
    df2['neg_integ_rate'] = -1*(df2['neg_integrated_events'] )/(df2['obs_length']*T)
    
    use2 = [x for x in np.unique(df2['exp_stage']) if 'washout' in x]
    
    
    plot_washout_summary(df2,use2,figsave,filetype,redo_stats = redo_stats,key = 'neg_event_rate', function = np.mean,function_name = 'np.mean',scale = 3, density = True)
    plot_washout_summary(df2,use2,figsave,filetype,redo_stats = False,key = 'neg_event_rate', function = np.mean,function_name = 'np.mean',scale = 3, density = False)
    #plot_washout_summary(df2,use2,figsave,filetype,redo_stats = redo_stats,key = 'neg_integ_rate', function = np.mean,function_name = 'np.mean',scale = 3, density = True)
    #plot_washout_summary(df2,use2,figsave,filetype,redo_stats = False,key = 'neg_integ_rate', function = np.mean,function_name = 'np.mean',scale = 3, density = False)
    
      

def plot_washout_summary(df,use,figsave,filetype,redo_stats = True,num_resamplings = 10**6,key = 'neg_event_rate', function = np.mean,function_name = 'np.mean',scale = 3, density = True):
    dfn = df.copy()     
        
    use_bool = np.array([np.any(x in use) for x in dfn.exp_stage])
    dfn = dfn[use_bool]
    
    pre = dfn[dfn.stage == 'pre'][key].to_numpy()
    post = dfn[dfn.stage == 'post'][key].to_numpy()
    wash = dfn[dfn.stage == 'washout'][key].to_numpy()   
    
    ppre = dfn[dfn.stage == 'pre'][[key,'day_slip']]
    ppost = dfn[dfn.stage == 'post'][[key,'day_slip']]
    wwash = dfn[dfn.stage == 'washout'][[key,'day_slip']]
    
    bins = np.histogram(np.concatenate((pre,post,wash))*10**3,bins = 10)[1]
    

    fig,axarr = plt.subplots(nrows = 3)
    c = 0.05
    axarr[0].hist(pre*10**scale,bins = bins, log = True, density = density, label = 'pre TTX', color = (c,c,c))
    axarr[1].hist(post*10**scale,bins = bins, log = True,  density = density, label = 'post 10 uM TTX', color = (c,c,c))
    axarr[2].hist(wash*10**scale,bins = bins, log = True,  density = density, label = 'washout', color = (c,c,c))
    axarr[0].sharey(axarr[1])
    axarr[2].sharey(axarr[1])
    
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
        
    
    fig.savefig(Path(figsave,'summary',f'TTX_washout_compare_density_{density}_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

    if redo_stats:
        p_pre_post,_,f1 = statsf.bootstrap_test(pre,post,function = function,plot = True,num_resamplings = num_resamplings, names = ['Pre TTX', 'Post TTX'])
        p_pre_wash,_,f2 = statsf.bootstrap_test_2sided(wash,pre,function = function,plot = True,num_resamplings = num_resamplings, names = ['Pre TTX', 'washout'])
        p_wash_post,_,f3 = statsf.bootstrap_test(wash,post,function = function,plot = True,num_resamplings = num_resamplings, names = ['Washout', 'Post TTX'])
        
        f1.savefig(Path(figsave,'summary','bootstrap',f'bootstrap_pre_post_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        f2.savefig(Path(figsave,'summary','bootstrap',f'bootstrap_wash_pre_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        f3.savefig(Path(figsave,'summary','bootstrap',f'bootstrap_wash_post_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        
        
        with open(Path(figsave, 'summary',f'statistical_test_results_washout_{key}.txt'),'w') as f:
            f.write(f'{datetime.datetime.now()}\n')
            f.write(f'Testing significance of second less than first for function {function_name}\n')
            f.write(f'N cells pre: {len(pre)}\n')
            f.write(f'N cells post: {len(post)}\n')
            f.write(f'N cells wash: {len(wash)}\n')
            f.write(f'N slips pre: {len(np.unique(ppre["day_slip"]))}\n')
            f.write(f'N slips post: {len(np.unique(ppost["day_slip"]))}\n')
            f.write(f'N slips wash: {len(np.unique(wwash["day_slip"]))}\n')
            
            f.write(f'Pre mean rate: {np.mean(pre)}\n')
            f.write(f'Post mean rate: {np.mean(post)}\n')
            f.write(f'Wash mean rate: {np.mean(wash)}\n')
    
            f.write(f'Num resamples: {num_resamplings}\n')
            f.write(f'p pre-post {p_pre_post}\n')
            f.write(f'p pre-wash (2 sided) {p_pre_wash}\n')
            f.write(f'p wash-post {p_wash_post}\n')
    
    

def plot_TTX_summary(df,use,figsave,filetype,redo_stats = True,num_resamplings = 10**6,key = 'neg_event_rate', function = np.mean,function_name = 'np.mean',scale = 3, density = True):
    dfn = df.copy()     
        
    use_bool = np.array([np.any(x in use) for x in dfn.exp_stage])
    dfn = dfn[use_bool]
    
    pre_10 = dfn[dfn.exp_stage == 'TTX_10um_pre'][key].to_numpy()
    post_10 = dfn[dfn.exp_stage == 'TTX_10um_post'][key].to_numpy()
    pre_1 = dfn[dfn.exp_stage == 'TTX_1um_pre'][key].to_numpy()
    post_1 = dfn[dfn.exp_stage == 'TTX_1um_post'][key].to_numpy()
    
    
    ppre_10 = dfn[dfn.exp_stage == 'TTX_10um_pre'][[key,'day_slip']]
    ppost_10 = dfn[dfn.exp_stage == 'TTX_10um_post'][[key,'day_slip']]
    
    ppre_1 = dfn[dfn.exp_stage == 'TTX_1um_pre'][[key,'day_slip']]
    ppost_1 = dfn[dfn.exp_stage == 'TTX_1um_post'][[key,'day_slip']]

    
    bins_10 = np.histogram(np.concatenate((pre_10,post_10))*10**3,bins = 10)[1]
    bins_1 = np.histogram(np.concatenate((pre_1,post_1))*10**3,bins = 10)[1]

    fig_10,axarr_10 = plt.subplots(nrows = 2)
    c = 0.05
    axarr_10[0].hist(pre_10*10**scale,bins = bins_10, log = True, density = density, label = 'pre TTX', color = (c,c,c))
    axarr_10[1].hist(post_10*10**scale,bins = bins_10, log = True,  density = density, label = 'post 10 uM TTX', color = (c,c,c))
    axarr_10[0].sharey(axarr_10[1])
    
    for idx,a in enumerate(axarr_10):
        if not density:
            a.set_ylim([0.6,10**4.5])
            a.set_yticks(10**np.arange(0,4,3))
        a.legend(frameon = False,loc = (0.4,0.4),fontsize = 16)
        pf.set_all_fontsize(a, 16)
        if idx != 1:
            a.set_xticklabels([])
            
        
    if not density:
        axarr_10[1].set_ylabel('Number of cells')
    else:
        axarr_10[1].set_ylabel('Proportion of cells')

    if key == 'neg_event_rate':
        axarr_10[-1].set_xlabel('Negative event rate ' + '(1000 cells$^{-1}$ s$^{-1}$)')
    elif key == 'neg_integ_rate':
        axarr_10[-1].set_xlabel(f'Integrated event rate per {10**scale} cells ' + '(%$\cdot$s / s)')
    else:
        raise ValueError('wrong key')
        
    
    fig_10.savefig(Path(figsave,'summary',f'TTX_10um_compare_density_{density}_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

    if redo_stats:
        p_pre_post_10,_,f1 = statsf.bootstrap_test(pre_10,post_10,function = function,plot = True,num_resamplings = num_resamplings, names = ['Pre TTX', 'Post 10 uM TTX'])
        f1.savefig(Path(figsave,'summary','bootstrap',f'bootstrap_pre_10_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        with open(Path(figsave,'summary', f'statistical_test_results_10uM_{key}.txt'),'w') as f:
            f.write(f'{datetime.datetime.now()}\n')
            f.write(f'Testing significance of second less than first for function {function_name}\n')
            f.write(f'N cells pre: {len(pre_10)}\n')
            f.write(f'N cells post: {len(post_10)}\n')
            f.write(f'N slips pre: {len(np.unique(ppre_10["day_slip"]))}\n')
            f.write(f'N slips post: {len(np.unique(ppost_10["day_slip"]))}\n')
            
            f.write(f'Pre mean rate: {np.mean(pre_10)}\n')
            f.write(f'Post mean rate: {np.mean(post_10)}\n')
            
            print('Hello')

    
            f.write(f'Num resamples: {num_resamplings}\n')
            f.write(f'p pre-post {p_pre_post_10}\n')
            
            
    fig_1,axarr_1 = plt.subplots(nrows = 2)
    c = 0.05
    axarr_1[0].hist(pre_1*10**scale,bins = bins_1, log = True, density = density, label = 'pre TTX', color = (c,c,c))
    axarr_1[1].hist(post_1*10**scale,bins = bins_1, log = True,  density = density, label = 'post 1 uM TTX', color = (c,c,c))
    axarr_1[0].sharey(axarr_1[1])

    
    for idx,a in enumerate(axarr_1):
        if not density:
            a.set_ylim([0.6,10**4.5])
            a.set_yticks(10**np.arange(0,4,3))
        a.legend(frameon = False,loc = (0.4,0.4),fontsize = 16)
        pf.set_all_fontsize(a, 16)
        if idx != 1:
            a.set_xticklabels([])
            
        
    if not density:
        axarr_1[1].set_ylabel('Number of cells')
    else:
        axarr_1[1].set_ylabel('Proportion of cells')

    if key == 'neg_event_rate':
        axarr_1[-1].set_xlabel('Negative event rate ' + '(1000 cells$^{-1}$ s$^{-1}$)')
    elif key == 'neg_integ_rate':
        axarr_1[-1].set_xlabel(f'Integrated event rate per {10**scale} cells ' + '(%$\cdot$s / s)')
    else:
        raise ValueError('wrong key')
        
    
    fig_1.savefig(Path(figsave,'summary',f'TTX_1um_compare_density_{density}_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

    if redo_stats:
        p_pre_post_1,_,f1 = statsf.bootstrap_test(pre_1,post_1,function = function,plot = True,num_resamplings = num_resamplings, names = ['Pre TTX', 'Post 1 uM TTX'])
        f1.savefig(Path(figsave,'summary','bootstrap',f'bootstrap_pre_1_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        with open(Path(figsave,'summary', f'statistical_test_results_1uM_{key}.txt'),'w') as f:
            f.write(f'{datetime.datetime.now()}\n')
            f.write(f'Testing significance of second less than first for function {function_name}\n')
            f.write(f'N cells pre: {len(pre_1)}\n')
            f.write(f'N cells post: {len(post_1)}\n')
            f.write(f'N slips pre: {len(np.unique(ppre_1["day_slip"]))}\n')
            f.write(f'N slips post: {len(np.unique(ppost_1["day_slip"]))}\n')
            
            f.write(f'Pre mean rate: {np.mean(pre_1)}\n')
            f.write(f'Post mean rate: {np.mean(post_1)}\n')

    
            f.write(f'Num resamples: {num_resamplings}\n')
            f.write(f'p pre-post {p_pre_post_1}\n')


    


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
    
    if only_neg:
        amp_bins = np.histogram(np.abs(dfn['event_amplitude'])*100,bins = nbins)[1]
    else:
        amp_bins = np.histogram(dfn['event_amplitude']*100,bins = nbins)[1]
    
    neg = dfn[dfn.stage == 'pre']
    pos = dfn[dfn.stage == 'post']
    
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.3, hspace=0.3) 
    
    fig,axarr = plt.subplots(figsize = (8,6))
    
    ax0 = plt.subplot(gs[0])
    if only_neg:
        ax0.hist(np.abs(neg['event_amplitude'])*100,bins = amp_bins,log = log,label = 'Pre',histtype = histtype)
        ax0.hist(np.abs(pos['event_amplitude'])*100,bins = amp_bins,log = log,label = f'TTX {TTX_level}'+' $\mathrm{\mu}$M',histtype = histtype)
    else:
        ax0.hist(neg['event_amplitude']*100,bins = amp_bins,log = log,label = 'Pre',histtype = histtype)
        ax0.hist(pos['event_amplitude']*100,bins = amp_bins,log = log,label = f'TTX {TTX_level}'+' $\mathrm{\mu}$M',histtype = histtype)
    ax0.set_xlabel('Absolute event amplitude (% $\Delta$R/R$_0$)')
    ax0.set_ylabel('Observed Frequency')    
    ax0.legend(frameon = False)
    
    ax1 = plt.subplot(gs[1])
    ax1.hist(np.abs(neg['event_length'])*T,bins = length_bins,log = log,label = 'Pre',histtype = histtype)
    ax1.hist(np.abs(pos['event_length'])*T,bins = length_bins,log = log,label = f'TTX {TTX_level}'+' $\mathrm{\mu}$M',histtype = histtype)
    ax1.set_xlabel('Event duration (s)')
    ax1.set_ylabel('Observed Frequency')   
    ax1.legend(frameon = False)
    
    
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    
    ax2 = plt.subplot(gs[2])
    if only_neg:
        h = ax2.hist2d(np.abs(neg['event_amplitude'])*100,neg['event_length']*T,bins = (amp_bins,length_bins),norm = norm)
    else:
        h = ax2.hist2d(neg['event_amplitude']*100,neg['event_length']*T,bins = (amp_bins,length_bins),norm = norm)
    plt.colorbar(h[3])
    ax2.set_xlabel('Pre-TTX event amplitude (% $\Delta$R/R$_0$)')    
    ax2.set_ylabel('Event duration (s)')
    
    ax3 = plt.subplot(gs[3])
    if only_neg:
        h2 = ax3.hist2d(np.abs(pos['event_amplitude'])*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    else:
        h2 = ax3.hist2d(pos['event_amplitude']*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    plt.colorbar(h2[3])
    ax3.set_xlabel('Post-TTX event amplitude (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event duration (s)')
    
    #get number of events before/after TTX
    thresh = -2
    iid = np.argwhere(h[1] > thresh)[0][0]
    n_events_pre = np.sum(h[0][:iid,:])
    n_events_post = np.sum(h2[0][:iid,:])

    with open(Path('/home/peter/Dropbox/Papers/cancer/v2/ttx_figure','num_bigneg_events.txt'),'w') as f:
        f.write(f'{datetime.datetime.now()}\n')
        f.write(f'Number events in bins up to edge at {h[1][iid]:.3f} %\n')
        f.write(f'pre: {n_events_pre} \n')
        f.write(f'post: {n_events_post} \n')
    
    
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
        amp_bins = np.histogram(np.abs(dfn['event_amplitude'])*100,bins = nbins)[1]
        
    else:
        amp_bins = np.histogram(dfn['event_amplitude']*100,bins = nbins)[1]
    
    length_bins = np.histogram(dfn['event_length']*T,bins = nbins)[1]

    
    neg = dfn[dfn.stage == 'pre']
    pos = dfn[dfn.stage == 'post']
    wash = dfn[dfn.stage == 'washout']
    
    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=0.3, hspace=0.3) 
    
    fig,axarr = plt.subplots(figsize = (8,6))
    
    ax0 = plt.subplot(gs[0])
    if only_neg:
        ax0.hist(np.abs(neg['event_amplitude'])*100,bins = amp_bins,log = log,label = 'Pre', histtype = histtype)
        ax0.hist(np.abs(pos['event_amplitude'])*100,bins = amp_bins,log = log,label = 'TTX 10 $\mathrm{\mu}$M', histtype = histtype)
        ax0.hist(np.abs(wash['event_amplitude'])*100,bins = amp_bins,log = log,label = 'Washout', histtype = histtype)
    else:
        ax0.hist(neg['event_amplitude']*100,bins = amp_bins,log = log,label = 'Pre', histtype = histtype)
        ax0.hist(pos['event_amplitude']*100,bins = amp_bins,log = log,label = 'TTX 10 $\mathrm{\mu}$M', histtype = histtype)
        ax0.hist(wash['event_amplitude']*100,bins = amp_bins,log = log,label = 'Washout', histtype = histtype)
    ax0.set_xlabel('Absolute event amplitude (% $\Delta$R/R$_0$)')
    ax0.set_ylabel('Observed Frequency')    
    ax0.legend(frameon = False)
    
    ax1 = plt.subplot(gs[1])
    ax1.hist(np.abs(neg['event_length'])*T,bins = length_bins,log = log,label = 'Pre', histtype = histtype)
    ax1.hist(np.abs(pos['event_length'])*T,bins = length_bins,log = log,label = 'TTX 10 $\mathrm{\mu}$M', histtype = histtype)
    ax1.hist(np.abs(wash['event_length'])*T,bins = length_bins,log = log,label = 'Washout', histtype = histtype)
    ax1.set_xlabel('Event duration (s)')
    ax1.set_ylabel('Observed Frequency')   
    ax1.legend(frameon = False)
    
    
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    
    ax2 = plt.subplot(gs[3])
    if only_neg:
        h = ax2.hist2d(np.abs(neg['event_amplitude'])*100,neg['event_length']*T,bins = (amp_bins,length_bins),norm = norm)
        
    else:
        h = ax2.hist2d(neg['event_amplitude']*100,neg['event_length']*T,bins = (amp_bins,length_bins),norm = norm)
    plt.colorbar(h[3])
    ax2.set_xlabel('Pre-TTX event amplitude (% $\Delta$R/R$_0$)')    
    ax2.set_ylabel('Event duration (s)')
    
    ax3 = plt.subplot(gs[4])
    if only_neg:
        h2 = ax3.hist2d(np.abs(pos['event_amplitude'])*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    else:
        h2 = ax3.hist2d(pos['event_amplitude']*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    plt.colorbar(h2[3])
    ax3.set_xlabel('Post-TTX Event amplitude (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event duration (s)')
    
    ax3 = plt.subplot(gs[5])
    if only_neg:
        h3 = ax3.hist2d(np.abs(wash['event_amplitude'])*100,wash['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    else:
        h3 = ax3.hist2d(wash['event_amplitude']*100,wash['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    plt.colorbar(h3[3])
    ax3.set_xlabel('Washout Event amplitude (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event duration (s)')
    
    #get number of events before/after TTX
    thresh = -2
    iid = np.argwhere(h[1] > thresh)[0][0]
    n_events_pre = np.sum(h[0][:iid,:])
    n_events_post = np.sum(h2[0][:iid,:])
    n_events_wash = np.sum(h3[0][:iid,:])
    
    return fig

if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    make_figures(initial_df,save_dir,figure_dir, redo_stats = False)