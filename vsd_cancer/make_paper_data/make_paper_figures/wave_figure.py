#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:33:36 2021

@author: peter
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from vsd_cancer.functions import correlation_functions as corrf

import f.plotting_functions as pf

def make_figures(top_dir,save_dir,figure_dir,filetype = '.png'):
    figsave = Path(figure_dir,'wave_figure')
    if not figsave.is_dir():
        figsave.mkdir()
    
    
    plot_correlation_analysis(top_dir,save_dir,figsave,filetype)
    




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


if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    make_figures(top_dir,save_dir,figure_dir)