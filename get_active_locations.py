#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:41:29 2020

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import pdb
import scipy.ndimage as ndimage
import pyqtgraph
import matplotlib.cm
import tifffile

import f.image_functions as imf
import f.plotting_functions as pf
import cancer_functions as canf
import f.general_functions as gf

def make_overlay(arr,stack,mask,cmap = matplotlib.cm.hot,alpha_top = 0.7,percent = 5,contrast = [0,0]):
    #mask
    
    overlay = cmap(arr)    
    overlay[...,-1] *= (~mask).astype(int)*alpha_top
    underlay = gf.to_8_bit(matplotlib.cm.Greys_r(stack))
    
    
    final = pf.alpha_composite(gf.to_8_bit(overlay), underlay)
    
    return final

def chunk_overlay(arr,norm_stack,chunk_size,cmap = matplotlib.cm.hot,alpha_top = 0.7,percent = 5, contrast = [0,0]):
    res = np.zeros(arr.shape + (4,),dtype = np.uint8)
    n_chunks,rem = np.divmod(arr.shape[0],chunk_size)  
    mask = np.logical_and(arr < np.percentile(arr,100-percent),arr > np.percentile(arr,percent))
    #apply contrast adjustment
    ma,mi = np.percentile(arr,100-contrast[0]),np.percentile(arr,contrast[0])
    arr[arr > ma] = ma
    arr[arr < mi] = mi
    arr = gf.norm(arr)
    ma2,mi2 = np.percentile(norm_stack,100-contrast[1]),np.percentile(norm_stack,contrast[1])#
    norm_stack[norm_stack > ma2] = ma2
    norm_stack[norm_stack < mi2] = mi2
    norm_stack = gf.norm(norm_stack)
    for i in range(n_chunks):
        res[i*chunk_size:(i+1)*chunk_size,...] =  make_overlay(arr[i*chunk_size:(i+1)*chunk_size,...],
                            norm_stack[i*chunk_size:(i+1)*chunk_size,...],
                            mask[i*chunk_size:(i+1)*chunk_size,...],
                            cmap = cmap,alpha_top=alpha_top,percent = percent,contrast=contrast)
        
    if rem != 0:
        res[-rem:,...] = make_overlay(arr[-rem:,...], norm_stack[-rem:,...],mask[-rem:,...],cmap = cmap,alpha_top=alpha_top,percent = percent,contrast=contrast)
    return res

topdir = '/home/peter/data/Firefly/cancer/analysis/'

df = pd.read_csv(topdir + 'long_acqs_processed.csv')

df = df[df.Activity == 'y']

mmouse = []
for idx,data in enumerate(df.itertuples()):

    data_dir = Path(topdir,'ratio_stacks',data.trial_string) 
    
    if Path(topdir,'tif_viewing', f'{data.trial_string}_overlay_2.tif').is_file():
        continue


    if False:
        rat = np.load(Path(data_dir, f'{data.trial_string}_ratio_stack.npy'))[:,2:-2,2:-2]
        rat2 = ndimage.filters.gaussian_filter(rat,(3,2,2))
        rat2 = np.pad(rat2,((0,0),(2,2),(2,2)),mode = 'edge')
        np.save(Path(data_dir, f'{data.trial_string}_ratio_stack_filtered.npy'),rat2)
        tifffile.imsave(Path(topdir,'tif_viewing', f'{data.trial_string}_ratio_stack_filtered.tif'),gf.to_16_bit(rat2))
    else:
        rat2 = np.load(Path(data_dir, f'{data.trial_string}_ratio_stack_filtered.npy'))
    
    stack = tifffile.imread(data.tif_file)[::2,...]

    display = chunk_overlay(rat2,stack,100,cmap = matplotlib.cm.Spectral,alpha_top=0.2,percent = 50,contrast = [0.5,0.1])

    tifffile.imsave(Path(topdir,'tif_viewing', f'{data.trial_string}_overlay_2.tif'),display)

    im = np.load(Path(data_dir, f'{data.trial_string}_im.npy'))
    
    std_im = np.std(rat2,0)
    #app = pg.mkQApp()
    #pg.image(rat2)
    #app.exec_()

        
    try:
        rois  = np.load(Path(data_dir, f'{data.trial_string}_rois.npy'))
    except FileNotFoundError:
        rois = imf.get_cell_rois(std_im.T, 4)[0]   
        np.save(Path(data_dir, f'{data.trial_string}_rois.npy'),rois)
    
    #overlay = pf.make_colormap_rois(rois, matplotlib.cm.gist_rainbow)[0]
    
    #stack = tifffile.imread(data.tif_file)
    
    #t_courses = np.array([gf.t_course_from_roi(rat2, r) for r in rois])



