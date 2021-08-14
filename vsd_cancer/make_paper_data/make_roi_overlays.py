#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 09:12:11 2020

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import f.plotting_functions as pf
import f.general_functions as gf

from vsd_cancer.functions import cancer_functions as canf

import tifffile
import cv2

import scipy.ndimage as ndimage

def make_all_overlay(df_file,save_dir,viewing_dir,HPC_num = None):
    df = pd.read_csv(df_file)

    for idx,data in enumerate(df.itertuples()):
        if HPC_num is not None: #allows running in parallel on HPC
            if idx != HPC_num:
                continue
        
        parts = Path(data.tif_file).parts
        trial_string = '_'.join(parts[parts.index('cancer'):-1])
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
        
        masks = canf.lab2masks(seg)
        
        im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
        try:
            overlay,colours = pf.make_colormap_rois(masks, 'gist_rainbow')
    
        except ValueError:
            continue
        
        fig,ax = plt.subplots(ncols = 3)
        ax[0].imshow(gf.norm(im),vmax = 0.6,cmap = 'Greys_r')
        ax[1].imshow(gf.norm(im),vmax = 0.6,cmap = 'Greys_r')
        ax[1].imshow(overlay)
        ax[2].imshow(gf.norm(im),vmax = 0.6,cmap = 'Greys_r')
        ax[2].imshow(overlay)
        pf.label_roi_centroids(ax[2], masks, colours,fontdict = {'fontsize' : 3})
        for a in ax:
            a.axis('off')
        ax[1].set_title(trial_string[:trial_string.find('long_acq')])
        fig.savefig(Path(viewing_dir,f'{trial_string}_rois.png'),bbox_inches = 'tight',dpi = 300)

        tif_overlay = (255*(np.sum(overlay,-1)>0)).astype(np.uint8)
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (100,100)
        fontScale              = 0.5
        fontColor              = 255
        lineType               = 2
        
        coms = [ndimage.center_of_mass(mask) for mask in masks]
        for idx,com in enumerate(coms):
            cv2.putText(tif_overlay,str(idx), 
                (int(com[1])-7,int(com[0])+7), 
                font, 
                fontScale,
                fontColor,
                lineType)
        
        tifffile.imsave(Path(viewing_dir,'../overlays/',f'{trial_string}_just_rois_withlab.tif'),tif_overlay)
        

