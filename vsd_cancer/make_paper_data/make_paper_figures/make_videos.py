#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 21:40:18 2021

@author: peter
"""


import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from vsd_cancer.functions import cancer_functions as canf

import tifffile

import pandas as pd

import scipy.ndimage as ndimage

import f.general_functions as gf


def make_all_videos(initial_df, save_dir,figure_dir, redo = True):
    
    figsave = Path(figure_dir,'videos')
    

    example_trial = 'cancer_20201207_slip1_area1_long_acq_corr_corr_long_acqu_blue_0.03465_green_0.07063_heated_to_37_1'
    make_video_parts(example_trial,initial_df, save_dir,figsave,downsample = 2, start = 0, stop = None, cells = None)
    
    raise ValueError('Gah not working yet!')
    wave_trial = 'cancer_20201216_slip1_area2_long_acq_long_acq_blue_0.0296_green_0.0765_heated_to_37_1'
    make_video_parts(wave_trial,initial_df, save_dir,figsave,downsample = 2, start = 4000, stop = 5000, cells = None)
    
    

def make_video_parts(trial_string,initial_df, save_dir,figsave,downsample = 5, start = 0, stop = None, cells = None):
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    seg =  np.load(Path(trial_save,f'{trial_string}_seg.npy'))
    masks = canf.lab2masks(seg)
    
    df = pd.read_csv(initial_df)
    df = df[df.trial_string == trial_string]
    
    rat = np.load(Path(trial_save,f'{trial_string}_ratio_stack.npy'))
    rat2 =ndimage.gaussian_filter(rat,(3,2,2))

    st = tifffile.imread(df.tif_file.iloc[0])[::2]
    
    if start != 0:
        st = st[start:]
        rat2 = rat2[start:]
        
    if stop is not None:
        st = st[:stop]
        rat2 = rat2[:stop]
        
    st = st[::downsample]
    rat2 = rat2[::downsample]
    
    if cells is not None:
        raise ValueError()
    
    #color balance
    cmin = np.percentile(rat2,0.05)
    cmax = np.percentile(rat2,99.95)
    rat2[np.where(rat2<cmin)] = cmin
    rat2[np.where(rat2>cmax)] = cmax
    rat2 = gf.to_8_bit(rat2)
    
    #color balance
    cmin = np.percentile(st,0.05)
    cmax = np.percentile(st,99.95)
    st[np.where(st<cmin)] = cmin
    st[np.where(st>cmax)] = cmax
    st = gf.to_8_bit(st)
    
    if not Path(figsave,trial_string).is_dir():
        Path(figsave,trial_string).mkdir(parents = True)
    
    tifffile.imsave(Path(figsave,trial_string,'stack.tif'),st)
    tifffile.imsave(Path(figsave,trial_string,'ratio.tif'),rat2)
  


if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    
    make_all_videos(initial_df, save_dir,figure_dir)
    