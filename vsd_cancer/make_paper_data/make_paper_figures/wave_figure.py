#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:33:36 2021

@author: peter
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import scipy.stats as stats

import scipy.ndimage as ndimage

from vsd_cancer.functions import correlation_functions as corrf

from vsd_cancer.functions import cancer_functions as canf

import pandas as pd

import f.plotting_functions as pf

def make_figures(top_dir,save_dir,figure_dir,filetype = '.png'):
    figsave = Path(figure_dir,'wave_figure')
    if not figsave.is_dir():
        figsave.mkdir()
    
    
    plot_correlation_analysis(top_dir,save_dir,figsave,filetype)
    plot_example_synchrony(top_dir,save_dir,figsave, filetype, redo_boot = False)
    plot_wave(top_dir,save_dir,figsave, filetype)




def plot_correlation_analysis(top_dir,save_dir,figsave, filetype):

    all_trains = np.load(Path(save_dir,'correlation','all_spike_trains.npy'),allow_pickle = True)
    
    
    CIs = np.load(Path(save_dir,'correlation','CIs.npy'))
    CIs_nd = np.load(Path(save_dir,'correlation','CIs_null.npy'))
    resamplings = np.load(Path(save_dir,'correlation','bootstrapped_samples.npy'))
    p_vals = np.load(Path(save_dir,'correlation','p_vals.npy'))
    null_dists = np.load(Path(save_dir,'correlation','null_dists.npy'))
    binsizes = np.load(Path(save_dir,'correlation','binsizes.npy'))
    
    
    means = [np.mean(corrf.get_all_pairwise(all_trains,binsize)) for binsize in binsizes]
    means_null = [np.mean(x) for x in null_dists]
    
    
    fig,ax = plt.subplots()
    ax.loglog(binsizes,means,'.-k',linewidth = 2, markersize = 10, label = 'Observed')
    for idx,c in enumerate(CIs):
        ax.loglog(binsizes[idx]*np.ones(2),c,'k')
        #ax.loglog(binsizes[idx] + np.array([-1,1]),c[0]*np.ones(2),'k')
        #ax.loglog(binsizes[idx] + np.array([-1,1]),c[1]*np.ones(2),'k')
    
    ax.loglog(binsizes,means_null,'.:k',linewidth = 2, markersize = 10, label = 'Shuffled')
    for idx,c in enumerate(CIs_nd):
        ax.loglog(binsizes[idx]*np.ones(2),c,'k')
        #ax.loglog(binsizes[idx] + np.array([-1,1]),c[0]*np.ones(2),'k')
        #ax.loglog(binsizes[idx] + np.array([-1,1]),c[1]*np.ones(2),'k')
        
    ax.set_xlabel('log(Correlation coefficient)')
    ax.set_ylabel('log(Bin size)')
    plt.legend(frameon = False)
    pf.set_thickaxes(ax, 3)
    pf.set_all_fontsize(ax, 16)
    
    fig.savefig(Path(figsave,f'Correlation_binchange{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)

def plot_example_synchrony(top_dir,save_dir,figsave, filetype, redo_boot = False):
    trial_string_use = 'cancer_20201207_slip1_area1_long_acq_corr_corr_long_acqu_blue_0.03465_green_0.07063_heated_to_37_1'
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    df = pd.read_csv(initial_df)
    num_traces = 15
    sep = 25
    
    traces = np.array([0,2,3,4,6,8])
    
    
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
    event_props = events['event_props']
    
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
    
    
    tcs = tcs[traces,:]
    so = so[traces]
    masks = masks[traces,:]
    
    tc_filt = ndimage.gaussian_filter(tcs,(0,3))#np.array([prox_tv.tv1_1d(t,0.01) for t in tcs])
    
    
    
    cmap = matplotlib.cm.viridis
    
    
    
    
    fig = plt.figure(constrained_layout = True)
    gs  = fig.add_gridspec(2,5)
    ax = fig.add_subplot(gs[:,-2:])
    colors = []
    
    event_times = []
    for i in range(len(traces)):
        #ax.plot([0,tcs.shape[-1]*T],np.ones(2)*i*100/sep,'k',alpha = 0.5)
        line = ax.plot(np.arange(tcs.shape[-1])*T,(tcs[i]-1)*100 + i*100/sep, color = cmap(i/len(traces)))
        _ = ax.plot(np.arange(tcs.shape[-1])*T,(tc_filt[i]-1)*100 + i*100/sep, color = 'k')
        colors.append(line[0].get_c())
        ev = events[so[i]]
        evp = event_props[so[i]]
        ax.text(-10,(i-0.15)*100/sep,f'{i}',fontdict = {'fontsize':14},color = colors[i],ha = 'right',va = 'center')
        if True:
            event_times.append([])
            for edx,l in enumerate(ev.T):
                if evp[edx][-1] < 0:
                    event_times[-1].append(np.mean(l)*T)
                    #ax.fill_betweenx([(i-0.5*0.9)*100/sep,(i+0.5*0.9)*100/sep],l[0]*T,l[1]*T,facecolor = 'r',alpha = 0.5)
    
    plt.axis('off')
    pf.plot_scalebar(ax, 0, (tcs[:num_traces].min()-1)*100, 200,3,thickness = 3)
    
    colors = (np.array(colors)*255).astype(np.uint8)
    #colors = np.hstack([colors,np.ones((colors.shape[0],1))])
    
    over = masks[:len(traces)]
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
    
    fig.savefig(Path(figsave,f'example_traces{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
    
    
    
    event_dict = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
    events = event_dict['events'][idx]
    event_props = events['event_props']
    
    keys = [x for x in events.keys() if type(x) != str]
    event_times2 = []
    for k in keys:
        event_times2.append([])
        ev = events[k]
        evp = event_props[k]
        for edx,l in enumerate(ev.T):
                if evp[edx][-1] < 0:
                    event_times2[-1].append(np.mean(l)*T)
                    
    event_times2 = [x for x in event_times2 if x !=[]]
    
    binned_spikes = corrf.bin_times(event_times2,10)
    binned_spikes_shuffled = np.copy(binned_spikes)
    [np.random.shuffle(x) for x in binned_spikes_shuffled]
    
    fig,axarr = plt.subplots(nrows = 2)
    axarr[0].imshow(binned_spikes,cmap = 'Greys')
    axarr[0].axis('auto')
    axarr[1].sharex(axarr[0])
    axarr[1].imshow(binned_spikes_shuffled,cmap = 'Greys')
    axarr[1].axis('auto')
    axarr[1].set_xlabel('Bin (10s)')
   
    axarr[0].set_ylabel('Cell')
    
    fig.savefig(Path(figsave,f'example_spike_trains{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
    
    
    if redo_boot:
        #now get an example bootstrap sampling
        pairwise_true = np.mean(corrf.calculate_pairwise_corrs(binned_spikes))
        pairwise_shuffled = []
        for i in range(10**4):
            [np.random.shuffle(x) for x in binned_spikes_shuffled]
            pairwise_shuffled.append(np.mean(corrf.calculate_pairwise_corrs(binned_spikes_shuffled)))
        
        fig,ax = plt.subplots()
        h = ax.hist(pairwise_shuffled,bins = 20)
        ax.plot([pairwise_true,pairwise_true],[0,h[0].max()], '-.k', linewidth = 3)
        
        fig.savefig(Path(figsave,f'example_null_vs_corr{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)


def plot_wave(top_dir,save_dir,figsave, filetype):
    
    initial_df = Path(top_dir,'analysis','long_acqs_20201230_experiments_correct.csv')

    df = pd.read_csv(initial_df)
    for idx,data in enumerate(df.itertuples()):
        if '20201216_slip1_area2_long_acq' in data.trial_string:
            break
        
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    
    
    im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
    seg =  np.load(Path(trial_save,f'{trial_string}_seg.npy'))
    
    
    
    masks = canf.lab2masks(seg)
    
    T = 0.2
    
    
    
    
    
    tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))[:,-1000:-400]
    tc_filt=ndimage.gaussian_filter(tcs,(0,3))
    
    loc = 300,550
    
    ac = tc_filt[:,loc[0]:loc[1]]  < 0.98
    
    roi_ids = np.where(np.sum(ac,-1) >0)[0]
    wh = np.where(ac)
    start_ids = wh[1][np.unique(wh[0],return_index = True)[1]]
    
    sort = np.argsort(start_ids)
    roi_ids = roi_ids[sort]
    start_ids = start_ids[sort]
    
    roi_t = tcs[roi_ids,:]
    roi_t_filt = ndimage.gaussian_filter(roi_t,(0,1))
    
    #now plot the dist/time stuff
    roi_masks = masks[roi_ids,...]
    coms = np.array([ndimage.center_of_mass(mask) for mask in roi_masks])
    
    dists = np.sqrt((coms[:,0] - coms[0,0])**2 + (coms[:,1] - coms[0,1])**2)*1.04
    times = np.argmin(roi_t_filt,axis = -1)*T
    times -= times[0]
    
    sizes = np.sum(np.abs(roi_t[:,loc[0]:loc[1]]),-1)
    sizes = np.min(roi_t[:,loc[0]:loc[1]],-1)
    sizes -= np.mean(roi_t[:,loc[0]-20:loc[0]])
    sizes *= 100
    
    fit = stats.linregress(times,dists)
    
    fig,ax = plt.subplots()
    ax.plot(times,fit.slope*times +fit.intercept,'k',linewidth = 2)
    ax.plot(times,dists,'.b',markersize = 10,label = 'Cell event peaks')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance ($\mathrm{\mu}$m)')
    ax.set_xticks([0,5,10,15])
    ax.set_yticks([0,250,500])
    ax.text(0,400,f'Fit speed: {fit.slope:.0f}'+' $\mathrm{\mu}$m/s\nr$^2$ ='+f' {fit.rvalue**2:.2f}')
    plt.legend(frameon = False, loc = (0,0.7))
    pf.set_thickaxes(ax, 3)
    pf.make_square_plot(ax)
    pf.set_all_fontsize(ax, 14)
    fig.savefig(Path(figsave,'../wave_figure2',f'speed_fit{filetype}'))
    
    #plot distance/magnitude
    fit2 = stats.linregress(dists,sizes)
    fig,ax = plt.subplots()
    ax.plot(dists,sizes,'.b',markersize = 10,label = 'Cell peak amplitudes')
    ax.set_yticks([-6,-3,0])
    ax.set_xticks([0,250,500])
    ax.set_ylabel('Transient peak amplitude (%)')
    ax.set_xlabel('Distance ($\mathrm{\mu}$m)')
    pf.set_thickaxes(ax, 3)
    pf.make_square_plot(ax)
    pf.set_all_fontsize(ax, 14)
    fig.savefig(Path(figsave,'../wave_figure2',f'size_plot{filetype}'))
    
    
    sep = 20#
    norm_dist = dists/dists.max()
    num = roi_ids.shape[0]
    cmap = matplotlib.cm.viridis
    
    fig = plt.figure(constrained_layout = True)
    gs  = fig.add_gridspec(2,5)
    ax = fig.add_subplot(gs[:,-2:])
    colors = []
    for i in range(num):
        line = ax.plot(np.arange(roi_t.shape[-1])*T,(roi_t[i]-1)*100 + i*100/sep, color = cmap(i/num))
        line2 = ax.plot(np.arange(roi_t.shape[-1])*T,(roi_t_filt[i]-1)*100 + i*100/sep, color = 'k')
        colors.append(line[0].get_c())
        ax.text(-60,(i-0.15)*100/sep,f'{i}',fontdict = {'fontsize':24},color = colors[i])
    
    
    plt.axis('off')
    pf.plot_scalebar(ax, 0, (tcs[:num].min()-1)*100 - 3, 20,3,thickness = 3)
    
    colors = (np.array(colors)*255).astype(np.uint8)
    #colors = np.hstack([colors,np.ones((colors.shape[0],1))])
    
    over = roi_masks
    struct = np.zeros((3,3,3))
    struct[1,...] = 1
    over = np.logical_xor(ndimage.binary_dilation(over,structure = struct,iterations = 2),over).astype(int)
    over = np.sum(over[...,None]*colors[:,None,None,:],0).astype(np.uint8)
    length = int(100/1.04)
    
    over[-20:-15,10:10+length] = np.ones(4,dtype = np.uint8)*255
    
    ax1 = fig.add_subplot(gs[:,:-2])
    ax1.imshow(im,cmap = 'Greys_r')
    ax1.imshow(over)
    plt.axis('off')
    pf.label_roi_centroids(ax1, roi_masks, colors/255)
    
    fig.savefig(Path(figsave,'../wave_figure2',f'wave_time_courses{filetype}'),bbox_inches = 'tight',dpi = 300)
    
    ex_roi = 185
    ex_t = tcs[ex_roi]


if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    make_figures(top_dir,save_dir,figure_dir,filetype='.pdf')