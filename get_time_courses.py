#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:55:59 2020

@author: peter
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
import scipy.signal

import f.general_functions as gf
import f.plotting_functions as pf

topdir = '/home/peter/data/Firefly/cancer/analysis/'
df = pd.read_csv(topdir + 'long_acqs_processed.csv')


df = df[df.Activity == 'y']

def get_scalebar_overlay(im,px_sz,length,width = None,pos = None,color = None):
    overlay = np.zeros(im.shape + (4,))
    px_len = int(np.round(length/px_sz))
    if width is None:
        width = -int(np.round(im.shape[0]/40))
                    
    if pos is None:
        pos = -np.round(np.array(im.shape)/20).astype(int)
        
    if color is None:
        color = [1,1,1]
    overlay[width+pos[0]:pos[0],-px_len + pos[1]:pos[1],:] = [*color,1]
    return overlay

for idx,data in enumerate(df.itertuples()):
    if idx != 3:
        continue
    else:
        break
    
    roi_files = Path(topdir,'rois').glob(f'{data.trial_string}_roi_*.roi')
    
    im = np.load(Path(topdir,'ratio_stacks',data.trial_string,f'{data.trial_string}_im.npy'))
    
    im = np.pad(im[2:-2,2:-2],((2,2),(2,2)),mode = 'edge') #remove wierd pixel
    
    rois = np.array([gf.read_roi_file(f,im_dims = im.shape)[1] for f in roi_files])
    
    ratio = np.load(Path(topdir,'ratio_stacks',data.trial_string,f'{data.trial_string}_ratio_stack.npy'))
    
    LED = np.load(Path(topdir,'ratio_stacks',data.trial_string,f'{data.trial_string}_LED.npy'))[::10]
    plt.figure()
    plt.plot(LED[:1000])
    
    vcVm =  np.load(Path(topdir,'ratio_stacks',data.trial_string,f'{data.trial_string}_vcVm.npy'))[::100]
    plt.figure()
    plt.plot(vcVm)
    
    t_courses = 100*(np.array([gf.t_course_from_roi(ratio, r) for r in rois]) - 1)
    np.save(Path(topdir,'ratio_stacks',data.trial_string,f'{data.trial_string}_rois.npy'),rois)
    np.save(Path(topdir,'ratio_stacks',data.trial_string,f'{data.trial_string}_t_courses.npy'),t_courses)
    
    fil_t_courses = scipy.signal.medfilt(t_courses,(1,21))
    
    overlay,col = pf.make_colormap_rois(rois, 'gist_rainbow')
    
    fig,axarr = plt.subplots(nrows = 1, ncols = 2)
    axarr[0].imshow(im)
    axarr[0].imshow(overlay)
    pf.label_roi_centroids(axarr[0], rois, col,fontdict = {'fontsize':10})
    axarr[0].imshow(get_scalebar_overlay(im,0.26*4,100,color  = [1,1,1]))
    axarr[0].axis('off')
    
    for i,tc in enumerate(fil_t_courses):
        axarr[1].plot(tc - 3*np.arange(t_courses.shape[0])[i],color =col[i],linewidth = 0.75)
        axarr[1].text(-120, -3*np.arange(t_courses.shape[0])[i],str(i+1),color = col[i],fontdict = {'fontsize':20}, ha = 'center', va = 'center')
        axarr[1].axis('off')
        
    pf.plot_scalebar(axarr[1], 5150, axarr[1].axis()[2], -60*5, 1,thickness = 4)
        
    fig.savefig(Path(topdir,'tif_viewing','t_courses',f'{data.trial_string}_time_course_im.svg'),transparent = True,bbox_inches = 'tight',dpi = 300)
    
    plt.show()