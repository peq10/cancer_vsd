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

import scipy.ndimage as ndimage

from vsd_cancer.functions import stats_functions as statsf


from vsd_cancer.functions import cancer_functions as canf

def make_figures(initial_df,save_dir,figure_dir,filetype = '.png', redo_stats = False):
    figsave = Path(figure_dir,'10A_figure')
    if not figsave.is_dir():
        figsave.mkdir()
    
    plot_MCF_hist(save_dir,figsave,filetype)

    plot_compare_mda(save_dir,figsave,filetype , 'neg_event_rate', np.mean, 'np.mean', scale = 3, density = False, redo_stats =redo_stats)
    plot_compare_mda(save_dir,figsave,filetype , 'neg_event_rate', np.mean, 'np.mean', scale = 3, density = True, redo_stats = False)
    #plot_compare_mda(save_dir,figsave,filetype , 'neg_integ_rate', np.mean, 'np.mean', scale = 3, density = False, redo_stats = redo_stats)
    #plot_compare_mda(save_dir,figsave,filetype , 'neg_integ_rate', np.mean, 'np.mean', scale = 3, density = True, redo_stats = False)
    plot_example_and_tcs(save_dir,figsave, filetype)

def plot_example_and_tcs(save_dir,figsave, filetype):
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
        trial_save = Path(save_dir,'ratio_stacks',t)
        
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
        
        fig.savefig(Path(figsave,'examples',f'MCF_TGF_Examples_{idx}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

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
    
    
    bins = np.histogram(np.concatenate((md,mc,tg))*10**3,bins = 10)[1]
    

    fig,axarr = plt.subplots(nrows = 3)
    c = 0.05
    axarr[0].hist(md*10**scale,bins = bins, log = True, density = density, label = 'MDA-MB-231', color = (c,c,c))
    axarr[1].hist(mc*10**scale,bins = bins, log = True,  density = density, label = 'MCF10A', color = (c,c,c))
    axarr[2].hist(tg*10**scale,bins = bins, log = True,  density = density, label = 'MCF10A+TGF$\\beta$', color = (c,c,c))
    
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
        
    fig.savefig(Path(figsave,'summary', f'MCF_compare_density_{density}_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

    if redo_stats:
        p_mda_mcf,_,f1 = statsf.bootstrap_test(md,mc,function = function,plot = True,num_resamplings = num_resamplings, names = ['MDA-MB-231', 'MCF10A'])
        p_mda_tgf,_,f2 = statsf.bootstrap_test(md,tg,function = function,plot = True,num_resamplings = num_resamplings, names = ['MDA-MB-231', 'MCF10A + TGF$\\beta$'])
        p_mcf_tgf,_,f3 = statsf.bootstrap_test(tg,mc,function = function,plot = True,num_resamplings = num_resamplings, names = ['MCF10A + TGF$\\beta$', 'MCF10A'])
        
        f1.savefig(Path(figsave,'bootstrap',f'bootstrap_231_MCF_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        f2.savefig(Path(figsave,'bootstrap',f'bootstrap_231_tgf_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        f3.savefig(Path(figsave,'bootstrap',f'bootstrap_tgf_MCF_{key}{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
        
        
        with open(Path(figsave,'summary', f'statistical_test_results_{key}.txt'),'w') as f:
            f.write(f'{datetime.datetime.now()}\n')
            f.write(f'Testing significance of second less than first for function {function_name}\n')
            f.write(f'N cells MDA: {len(md)}\n')
            f.write(f'N cells MCF: {len(mc)}\n')
            f.write(f'N cells TGF: {len(tg)}\n')
            f.write(f'N slips MDA: {len(np.unique(mda["day_slip"]))}\n')
            f.write(f'N slips MCF: {len(np.unique(mcf["day_slip"]))}\n')
            f.write(f'N slips TGF: {len(np.unique(tgf["day_slip"]))}\n')
            
            f.write(f'231 mean rate: {np.mean(md)}')
            f.write(f'MCF10A mean rate: {np.mean(mc)}')
            f.write(f'MCF10A + TGFB mean rate: {np.mean(tg)}')
    
            f.write(f'Num resamples: {num_resamplings}\n')
            f.write(f'p MDA-MCF {p_mda_mcf}\n')
            f.write(f'p MDA-TGF {p_mda_tgf}\n')
            f.write(f'p MCF-TGF {p_mcf_tgf}\n')


def plot_MCF_hist(save_dir,figsave,filetype):
    
    df = pd.read_csv(Path(save_dir,'all_events_df.csv'))
    df['exp_stage'] = df.expt + '_' + df.stage

    use = ['MCF10A_TGFB_none', 'MCF10A_none']
    
    

    log = [True]
    only_neg = [False]
    histtype = ['bar']

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
    ax1.set_xlabel('Event duration (s)')
    ax1.set_ylabel('Observed Frequency')   
    ax1.legend(frameon = False)
    
    
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    
    ax2 = plt.subplot(gs[2])
    h = ax2.hist2d(neg['event_amplitude']*100,neg['event_length']*T,bins = (amp_bins,length_bins),norm = norm)
    plt.colorbar(h[3])
    ax2.set_xlabel('MCF10A event amplitude (% $\Delta$R/R$_0$)')    
    ax2.set_ylabel('Event duration (s)')
    
    ax3 = plt.subplot(gs[3])
    h2 = ax3.hist2d(pos['event_amplitude']*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    plt.colorbar(h2[3])
    ax3.set_xlabel('MCF10A + TGF-$\\beta$ event amplitude (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event duration (s)')
    
    return fig

def look_at_diff_tgf_lengths(save_dir):
    
    T = 0.2
    df = pd.read_csv(Path(save_dir,'non_ttx_active_df_by_cell.csv'))
    key = 'neg_event_rate'
    df['active'] =  ((df.n_neg_events + df.n_pos_events) > 0).astype(int)
    df['prop_pos'] =  df.n_pos_events/(df.n_neg_events + df.n_pos_events)
    df['prop_neg'] =  df.n_neg_events/(df.n_neg_events + df.n_pos_events)
    df['prop_pos'] =  df.n_pos_events/(df.n_neg_events + df.n_pos_events)
    df['prop_neg'] =  df.n_neg_events/(df.n_neg_events + df.n_pos_events)
    df['day_slip'] = df.day.astype(str) + '_' + df.slip.astype(str)
    
    df['exp_stage'] = df.expt + '_' + df.stage
    df['day_slip'] = df.day.astype(str) + '_' + df.slip.astype(str) 
    
    df['neg_event_rate'] = (df['n_neg_events'] )/(df['obs_length']*0.2)
    
    
    tgf = df[df.exp_stage == 'MCF10A_TGFB_none']
    
    tgf_active = tgf[tgf.active == 1]


    

    #just use a lookup for the length of TGF treatment
    #from Mar email: I started to treat the cells with TGFb on sunday 7th March at 5pm
    tgf_lookup = {20210122: 48,20210312: 5*24, 20210313: 6*24, 20210314: 7*24}
    
    def sem(x):
        return np.std(x)/np.sqrt(len(x))
    
    prop_active_tgf = tgf[['active','neg_event_rate','day']].groupby(['day']).agg([np.mean, sem])

    rate_tgf_active = tgf_active[['neg_event_rate','day']].groupby(['day']).agg([np.mean, sem])


    
    fig,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()
    for data1,data2 in zip(prop_active_tgf.itertuples(), rate_tgf_active.itertuples()):
        x1 = tgf_lookup[data1.Index]
        x2 = tgf_lookup[data2.Index]
        mean_active = data1._1*100
        mean_rate = data2._1
        mean_rate_all = data1._3
        

        sem_active = data1._2*100
        sem_rate = data2._2
        sem_rate_all = data1._4
        
        ax1.errorbar(x1,mean_active, yerr = sem_active)
        ax2.errorbar(x2,mean_rate,yerr = sem_rate)
        ax3.errorbar(x2,mean_rate_all,yerr = sem_rate_all)
        
    ax1.set_xlabel('Time in TGFB')
    ax1.set_ylabel('% active cells')
    ax2.set_xlabel('Time in TGFB')
    ax2.set_ylabel('Mean neg event rate\n(active cells)')
    
    ax3.set_xlabel('Time in TGFB')
    ax3.set_ylabel('Mean neg event rate\n(all cells)')

    return 0

if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    make_figures(initial_df,save_dir,figure_dir, redo_stats = True)