#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:47:33 2021

@author: peter
"""

from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage

import matplotlib.markers
import matplotlib.pyplot as plt

import f.plotting_functions as pf

def make_figures(initial_df,save_dir,figure_dir,filetype = '.png'):
    
    figsave = Path(figure_dir,'method_figure')
    if not figsave.is_dir():
        figsave.mkdir()
    
    trial_string = 'cancer_20201215_slip2_area1_long_acq_corr_long_acq_blue_0.0296_green_0.0765_heated_to_37_1'
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    
    #now plot the example with surround
    cell = 91
    sep = 0
    detection = 0
    
    start,end = 100,780
    start *=5
    end *= 5
    
    
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))[cell,...]
    event_dict = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
    events = event_dict['events'][detection]
    
    
    tc = tc[start:end]

    T = 0.2
    eve = events[cell]
    tc_filt = ndimage.gaussian_filter(tc,3)
    t = np.arange(tc.shape[-1])*T

    fig,ax = plt.subplots()
    ax.plot(t,(tc-1)*100,'k',linewidth = 2, alpha = 0.5)
    #ax.plot(t,(tc_test-1)*100,'r',linewidth = 1)
    
    offset = -sep
    #ax.plot([0,t.max()],np.array([0,0])+offset,'r',linewidth = 1,alpha = 0.7)
    ax.plot(t,(tc_filt-1)*100+offset,'k',linewidth = 2)
    #axb.plot(t,(tc_test-1)*100,'k',linewidth = 2,alpha = 0.4)
    #ax.plot([0,t.max()],np.array([-1,1])[None,:]*np.array([thresh*100,thresh*100])[:,None]+offset,'--r',linewidth = 2)
    for l in eve.T:
        if l[0] > end or l[1] < start:
            continue
        #ax.fill_betweenx(np.array([tc_filt.min()-1+offset/(100*1.1),tc.max()-1])*1.1*100,(l[0]-start)*T,(l[1]-1-start)*T,facecolor = 'r',alpha = 0.5)
        ax.plot((np.mean(l) - start)*T,tc.max()*1.5,'r',marker = matplotlib.markers.CARETDOWN,markersize =8)
    plt.axis('off')
    pf.plot_scalebar(ax, 0, 1.5*(tc_filt.min()-1)*100+offset, 50,1,thickness = 3)
    
    fig.savefig(Path(figsave,f'example_tc{filetype}'),bbox_inches = 'tight',dpi = 300,transparent = True)
    





if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    make_figures(initial_df,save_dir,figure_dir)