#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:33:36 2021

@author: peter
"""
from pathlib import Path

from vsd_cancer.functions import cancer_functions as canf

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import f.plotting_functions as pf

import matplotlib.cm
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import seaborn as sns

def make_figures(initial_df,save_dir,figure_dir,filetype = '.png'):
    figsave = Path(figure_dir,'231_figure')
    if not figsave.is_dir():
        figsave.mkdir()
        
        
    trial_string_use = 'cancer_20201207_slip1_area1_long_acq_corr_corr_long_acqu_blue_0.03465_green_0.07063_heated_to_37_1'
    
    df = pd.read_csv(initial_df)
    
    
    if False:
        num_traces = 15
        sep = 25
        make_example_trace_fig(trial_string_use,num_traces,sep,df,save_dir,figsave,filetype)
        #make example for synchrony fig
        make_example_trace_fig(trial_string_use,15,sep,df,save_dir,Path(figure_dir,'wave_figure'),filetype)
        
        plot_positive_negative_events(save_dir,figsave,filetype)
        
    
    plot_percent_quiet(save_dir, figsave, filetype)
    
    
def plot_percent_quiet(save_dir,figsave,filetype):
    
    df = pd.read_csv(Path(save_dir,'non_ttx_active_df_by_cell.csv'))

    T = 0.2
    
    
    df['exp_stage'] = df.expt + '_' + df.stage
    df['day_slip'] = df.day.astype(str) + '_' + df.slip.astype(str) 
    
    
    df['event_rate'] = (df['n_neg_events'] +  df['n_pos_events'])/(df['obs_length']*T)
    df['neg_event_rate'] = (df['n_neg_events'] )/(df['obs_length']*T)
    
    df['integ_rate'] = (df['integrated_events'])/(df['obs_length']*T)
    df['neg_integ_rate'] = -1*(df['neg_integrated_events'] )/(df['obs_length']*T)
    
    mda = df[df.exp_stage == 'standard_none'][['neg_event_rate','day_slip']]
    
    mda['active'] = 100*(mda['neg_event_rate'] > 0).astype(int)
    
    
    active_d = mda[mda.active != 0]
    active_rate = active_d.groupby('day_slip').mean()['neg_event_rate'].to_numpy()
    
    active = mda.groupby('day_slip').mean()['active'].to_numpy()
    
    fig,ax4 = plt.subplots()
    #sns.violinplot(y=active,saturation = 0.5)
    #ax4.plot(np.random.normal(loc = 1,scale = scale,size = sens.shape[0]),sens[:,-1],'.k',markersize = 12)
    sns.swarmplot(y=active,ax = ax4,color = 'k',size = 7)
    ax4.xaxis.set_visible(False)
    ax4.set_ylabel('Active Cells (%)')
    pf.set_thickaxes(ax4, 3,remove = ['top','right','bottom'])
    pf.set_all_fontsize(ax4, 16)
    pf.set_all_fontsize(ax4, 16)
    
    pf.make_square_plot(ax4)
    fig.savefig(Path(figsave,f'231_active_cells{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
    
    
    fig,ax = plt.subplots()
    #sns.violinplot(y=active,saturation = 0.5)
    #ax4.plot(np.random.normal(loc = 1,scale = scale,size = sens.shape[0]),sens[:,-1],'.k',markersize = 12)
    sns.swarmplot(y=active_rate*10**3,ax = ax,color = 'k',size = 7)
    ax.xaxis.set_visible(False)
    ax.set_ylabel('Mean event rate\n(active cells, x10$^3$ s$^{-1}$)')
    pf.set_thickaxes(ax, 3,remove = ['top','right','bottom'])
    pf.set_all_fontsize(ax, 16)
    pf.set_all_fontsize(ax, 16)
    
    pf.make_square_plot(ax)
    fig.savefig(Path(figsave,f'231_active_rates{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
    
    fig1,ax1 = plt.subplots()
    ax1.hist(active_d.neg_event_rate*1000,bins = 20, log = True, color = (0.2,0.2,0.2))
    ax1.set_xlabel('Event rate(active cells, x10$^3$ s$^{-1}$)')
    ax1.set_ylabel('Number of cells')
    pf.set_thickaxes(ax1, 3)
    pf.set_all_fontsize(ax1, 16)
    pf.set_all_fontsize(ax1, 16)
    fig1.savefig(Path(figsave,f'231_active_hists{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

    
    
def make_example_trace_fig(trial_string_use,num_traces,sep,df,save_dir,figsave,filetype):
    T = 0.2

    im_scalebar_length_um = 100
    
    for idx,data in enumerate(df.itertuples()):
        trial_string = data.trial_string
        #print(trial_string)
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        if trial_string != trial_string_use:
            continue
        else:
            break

    im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
    seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
      
    masks = canf.lab2masks(seg)
    
    tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    
    event_dict = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

    idx = 0
    events = event_dict['events'][idx]
    
    keep = [x for x in np.arange(tcs.shape[0])]

    #sort by event amounts 
    sort_order = np.array([np.sum(np.abs(events['event_props'][x][:,-1])) if x in events.keys() else 0 for x in range(tcs.shape[0])])
    
    tcs = tcs[keep,:]
    masks = masks[keep,...]
    sort_order = np.argsort(sort_order[keep])[::-1]
    
    tcs = tcs[sort_order,:]
    masks = masks[sort_order,:]
    so = np.array(keep)[sort_order]
    

    
    tcs = tcs[:num_traces,...]
    so = so[:num_traces]
    masks = masks[:num_traces,...]
    
    #now sort back in position
    llocs = np.array([ndimage.measurements.center_of_mass(x) for x in masks])
    llocs = llocs[:,0]*masks.shape[-2] + llocs[:,1]
    order2 = np.argsort(llocs)[::-1]


    tcs = tcs[order2,...]
    so = so[order2,...]
    masks = masks[order2,...]
    
    tc_filt = ndimage.gaussian_filter(tcs,(0,3))#np.array([prox_tv.tv1_1d(t,0.01) for t in tcs])
    


    cmap = matplotlib.cm.viridis
    
    
    
    
    fig = plt.figure(constrained_layout = True)
    gs  = fig.add_gridspec(2,5)
    ax = fig.add_subplot(gs[:,-2:])
    colors = []
    for i in range(num_traces):
        ax.plot([0,tcs.shape[-1]*T],np.ones(2)*i*100/sep,'k',alpha = 0.5)
        line = ax.plot(np.arange(tcs.shape[-1])*T,(tcs[i]-1)*100 + i*100/sep, color = cmap(i/num_traces))
        _ = ax.plot(np.arange(tcs.shape[-1])*T,(tc_filt[i]-1)*100 + i*100/sep, color = 'k')
        colors.append(line[0].get_c())
        ev = events[so[i]]
        ax.text(-10,(i-0.15)*100/sep,f'{i}',fontdict = {'fontsize':14},color = colors[i],ha = 'right',va = 'center')
        if False:
            for l in ev.T:
                ax.fill_betweenx([(i-0.5*0.9)*100/sep,(i+0.5*0.9)*100/sep],l[0]*T,l[1]*T,facecolor = 'r',alpha = 0.5)
    
    plt.axis('off')
    pf.plot_scalebar(ax, 0, (tcs[:num_traces].min()-1)*100, 200,3,thickness = 3)
    
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
    
    fig.savefig(Path(figsave,f'example_tcs{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
    
def plot_positive_negative_events(save_dir,figsave,filetype):
    
    df = pd.read_csv(Path(save_dir,'all_events_df.csv'))
    df['exp_stage'] = df.expt + '_' + df.stage

    use = ['TTX_10um_washout_pre','TTX_1um_pre','TTX_10um_pre','standard_none']
    
    

    
    fig1 = plot_events(df,use,log = True)
    fig1.savefig(Path(figsave,f'231_histograms_log{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

    fig2 = plot_events(df,use,log = False)
    fig2.savefig(Path(figsave,f'231_histograms_no_log{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
    
    fig3 = plot_events2(df,use,log = True)
    fig3.savefig(Path(figsave,f'231_histograms_log_both_pos{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

    fig4 = plot_events2(df,use,log = False)
    fig4.savefig(Path(figsave,f'231_histograms_no_log_both_pos{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)





def plot_events(df,use,log = True,upper_lim = 6.6,lower_lim = 0, T = 0.2,nbins = 50, only_neg = True):
    
    
    dfn = df.copy()
    
    
    use_bool = np.array([np.any(x in use) for x in dfn.exp_stage])
    dfn = dfn[use_bool]
    

    too_big = np.abs(dfn.event_amplitude) > 6.6/100
    too_small =  np.abs(dfn.event_amplitude) < 0/100
    dfn = dfn[np.logical_not(np.logical_or(too_big,too_small))]
    
    length_bins = np.histogram(dfn['event_length']*T,bins = nbins)[1]
    
    
    
    amp_bins = np.histogram(np.abs(dfn['event_amplitude'])*100,bins = nbins)[1]
    
    neg = dfn[dfn.event_amplitude <0]
    pos = dfn[dfn.event_amplitude >0]
    
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.3, hspace=0.3) 
    
    fig,axarr = plt.subplots(figsize = (8,6))
    
    ax0 = plt.subplot(gs[0])
    ax0.hist(np.abs(neg['event_amplitude'])*100,bins = amp_bins,log = log,label = '-ve')
    ax0.hist(pos['event_amplitude']*100,bins = amp_bins,log = log,label = '+ve')
    ax0.set_xlabel('Absolute event amplitude (% $\Delta$R/R$_0$)')
    ax0.set_ylabel('Observed Frequency')    
    ax0.legend(frameon = False)
    
    ax1 = plt.subplot(gs[1])
    ax1.hist(np.abs(neg['event_length'])*T,bins = length_bins,log = log,label = '-ve')
    ax1.hist(pos['event_length']*T,bins = length_bins,log = log,label = '+ve')
    ax1.set_xlabel('Event length (s)')
    ax1.set_ylabel('Observed Frequency')   
    ax1.legend(frameon = False)
    
    
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    
    ax2 = plt.subplot(gs[2])
    ax2.hist2d(np.abs(neg['event_amplitude'])*100,neg['event_length']*T,bins = (amp_bins,length_bins),norm = norm)

    ax2.set_xlabel('Negative event amplitude (% $\Delta$R/R$_0$)')    
    ax2.set_ylabel('Event length (s)')
    
    ax3 = plt.subplot(gs[3])

    ax3.hist2d(np.abs(pos['event_amplitude'])*100,pos['event_length']*T,bins = (amp_bins,length_bins),norm=norm)
    ax3.set_xlabel('Positive event size (% $\Delta$R/R$_0$)')    
    ax3.set_ylabel('Event length (s)')
    
    
    

    
    return fig

def plot_events2(df,use,log = True,upper_lim = 6.6,lower_lim = 0, T = 0.2,nbins = 50, only_neg = True):
    
    
    dfn = df.copy()
    
    
    use_bool = np.array([np.any(x in use) for x in dfn.exp_stage])
    dfn = dfn[use_bool]
    

    too_big = np.abs(dfn.event_amplitude) > 6.6/100
    too_small =  np.abs(dfn.event_amplitude) < 0/100
    dfn = dfn[np.logical_not(np.logical_or(too_big,too_small))]
    
    length_bins = np.histogram(dfn['event_length']*T,bins = nbins)[1]
    
    
    
    amp_bins = np.histogram(dfn['event_amplitude']*100,bins = nbins)[1]
    
    neg = dfn[dfn.event_amplitude <0]
    pos = dfn[dfn.event_amplitude >0]
    
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.3, hspace=0.3) 
    
    fig,axarr = plt.subplots(figsize = (8,6))
    
    ax0 = plt.subplot(gs[0])
    ax0.hist(neg['event_amplitude']*100,bins = amp_bins,log = log,label = '-ve')
    ax0.hist(pos['event_amplitude']*100,bins = amp_bins,log = log,label = '+ve')
    ax0.set_xlabel('Event amplitude (% $\Delta$R/R$_0$)')
    ax0.set_ylabel('Observed Frequency')    
    ax0.legend(frameon = False)
    
    ax1 = plt.subplot(gs[1])
    ax1.hist(np.abs(neg['event_length'])*T,bins = length_bins,log = log,label = '-ve')
    ax1.hist(pos['event_length']*T,bins = length_bins,log = log,label = '+ve')
    ax1.set_xlabel('Event length (s)')
    ax1.set_ylabel('Observed Frequency')   
    ax1.legend(frameon = False)
    
    
    if log:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
    
    ax2 = plt.subplot(gs[2])
    ax2.hist2d(dfn['event_amplitude']*100,dfn['event_length']*T,bins = (amp_bins,length_bins),norm = norm)

    ax2.set_xlabel('Event amplitude (% $\Delta$R/R$_0$)')    
    ax2.set_ylabel('Event length (s)')
    

    
    
    

    
    return fig


if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    make_figures(initial_df,save_dir,figure_dir)