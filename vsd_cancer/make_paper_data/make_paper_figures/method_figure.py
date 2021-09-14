#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:47:33 2021

@author: peter
"""

from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage
import pandas as pd

import matplotlib.markers
import matplotlib.pyplot as plt

import f.plotting_functions as pf

def make_figures(initial_df,save_dir,figure_dir,filetype = '.png'):
    
    figsave = Path(figure_dir,'method_figure')
    if not figsave.is_dir():
        figsave.mkdir()
    
    make_example_trace(initial_df,save_dir,figsave,filetype)
    
    make_spectral_shift_trace(figsave,save_dir,filetype)
    
    
def make_spectral_shift_trace(figsave,save_dir,filetype,delta = 20,exc1 = 405,exc2 = 530):
    ex = pd.read_csv(Path(save_dir,'spectra_data/excitation.csv'),names = ['wl','ex'])
    ex = np.array([ex['wl'],ex['ex']]).T
    ex = ex[ex[:,1]<1.1,:]
    ex = ex[ex[:,0]<640,:]
    ex = ex[::2,:]
    
    em = pd.read_csv(Path(save_dir,'spectra_data/Emission.csv'),names = ['wl','em'])
    em = np.array([em['wl'],em['em']]).T
    em = em[em[:,1]<1.1,:]
    em = em[~np.logical_and(em[:,1]>0.47,em[:,0]>713),:]
    em = em[em[:,0]<798,:]
    em = em[::2,:]
    
    
    ex_plus = np.copy(ex)
    em_plus = np.copy(em)
    
    ex_plus[:,0] += delta
    em_plus[:,0] += delta
    
    ex_minus = np.copy(ex)
    em_minus = np.copy(em)
    
    ex_minus[:,0] -= delta
    em_minus[:,0] -= delta
    
    def plot_spec(ax,spec,lab,*args,**kwargs):
        return ax.plot(spec[:,0],spec[:,1],*args,label = lab,**kwargs)

    fig,ax = plt.subplots()
    plot_spec(ax,ex,'Exc. Spectrum','-b',linewidth = 3)
    plot_spec(ax,em,'Em. Spectrum','-r',linewidth = 3)
    #plot_spec(ax,f_650_150,'Em. Filter','-k',linewidth = 3)
    
    ax.plot([exc1,exc1],[0,1],'--k',linewidth = 3)
    ax.plot([exc2,exc2],[0,1],'--k',linewidth = 3)
    
    plot_spec(ax, ex_minus, 'Ex minus', '--b', linewidth = 3)
    plot_spec(ax, em_minus, 'Em minus', '--r', linewidth = 3)
    
    plot_spec(ax, ex_plus, 'Ex plus', '--b', linewidth = 3)
    plot_spec(ax, em_plus, 'Em plus', '--r', linewidth = 3)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Rel. Absorption/Emission')
    ax.set_xlim([380,800])
    
    #ax.legend(frameon = False,loc = 1,fontsize=16)
    
    pf.set_thickaxes(ax, 3)
    pf.set_all_fontsize(ax, 16)
    pf.make_square_plot(ax)
    
    fig.savefig(Path(figsave,f'regular_spectrum{filetype}'),bbox_inches = 'tight',transparent=True)
    #pf.make_square_plot(ax)



    
def make_example_trace(initial_df,save_dir,figsave,filetype):
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