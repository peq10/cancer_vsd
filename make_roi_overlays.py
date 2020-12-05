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


def lab2masks(seg):
    masks = []
    for i in range(1,seg.max()):
        masks.append((seg == i).astype(int))
    return np.array(masks)


def norm(arr):
    return (arr - arr.min())/(arr.max() - arr.min()) 

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
        
        masks = lab2masks(seg)
        
        im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
        try:
            overlay,colours = pf.make_colormap_rois(masks, 'gist_rainbow')
        except ValueError:
            continue
        
        fig,ax = plt.subplots()
        ax.imshow(norm(im),vmax = 0.6,cmap = 'Greys_r')
        ax.imshow(overlay)
        pf.label_roi_centroids(ax, masks, colours)
        plt.axis('off')
        fig.savefig(Path(viewing_dir,f'{trial_string}_rois.png'),bbox_inches = 'tight',dpi = 300)
        plt.show()
        

