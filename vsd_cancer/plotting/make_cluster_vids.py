#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:07:19 2021

@author: peter
"""
import numpy as np
from pathlib import Path
import pandas as pd

import scipy.ndimage as ndimage

import matplotlib.cm

import os
import tifffile
import time
import f.general_functions as gf 
import prox_tv
import pyqtgraph as pg

from vsd_cancer.functions import cancer_functions as canf

import cv2

def make_roi_overlay(events_dict,seg,sz):
    overlay = np.zeros(sz,dtype = int)
    for idx in events_dict.keys():
        if type(idx) == str:
            continue
        for idx2 in range(events_dict[idx].shape[-1]):
            ids = events_dict[idx][:,idx2]
            mask = (seg == idx + 1).astype(int)
            outline = np.logical_xor(mask,ndimage.binary_dilation(mask,iterations = 4)).astype(int)
            overlay[ids[0]:ids[1],...] += outline
            
    
    overlay = (overlay > 0)

    return overlay


def make_type_mask_overlay(type_masks,masks):
    overlay = np.zeros(masks.shape[-2:]+(4,),dtype = float)
    colors = {0:0,1:1,2:2,3:3,4:4,8:5}
    names = {0:'W',1:'N',2:'Bs',3:'Bl',4:'Q',8:'N/A'}
    for t in type_masks:
        color = (np.array(matplotlib.cm.Set1((colors[t[0]]/5)))*255).astype(np.uint8)
        ma = np.sum(masks[t[1],...],0)
        ma = np.logical_xor(ma,ndimage.binary_dilation(ma,iterations = 4)).astype(int)
        ma = ma[:,:,None]*color[None,:]
        
        print(color)
        
        overlay += ma
        
        txt = np.zeros(ma.shape[:2] +(4,),dtype = np.uint8)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10 + colors[t[0]]*40,ma.shape[1]-20)
        fontScale              = 1
        fontColor              = [int(x) for x in color]
        lineType               = 2
        txt = cv2.putText(txt,names[t[0]], bottomLeftCornerOfText, font, fontScale,fontColor, lineType)
        
        overlay += txt
        
    overlay[overlay > 255] = 255
    
    return overlay.astype(np.uint8)
    

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','grey_videos')
initial_df = Path(top_dir,'analysis','full',f'long_acqs_20210428_experiments_correct_loaded_long.csv')

df = pd.read_csv(initial_df)
roi_df = pd.read_csv(Path(save_dir,'roi_df.csv'))

yilin_df = pd.read_csv(Path(top_dir,'analysis/full/yilin_cell_label.csv'))


yilin_trials = np.unique(yilin_df.trial_string)
downsample = 5

for idx,data in enumerate(df.itertuples()):

    t0 = time.time()
    
    trial_string = data.trial_string
    
    if trial_string not in yilin_trials:
        continue


    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    print(trial_string)


    
    try:
        finish_at = int(data.finish_at)*5
    except ValueError:
        finish_at = None
        
     
    seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))

    
    masks = canf.lab2masks(seg)
    
    curr_labels = yilin_df[yilin_df.trial_string == trial_string][['id','type']]
    
    type_masks = []
    
    for t in np.unique(curr_labels['type']):
        ids = curr_labels[curr_labels.type == t]['id']
        type_masks.append((t,ids.values))
    

    
    rat2 = np.load(Path(trial_save, f'{data.trial_string}_ratio_stack.npy'))[:finish_at]
    rat2 =ndimage.gaussian_filter(rat2,(3,2,2))
    type_overlay = make_type_mask_overlay(type_masks,masks)
    
    downsample = 5

    
    rat2 = rat2[::downsample]
    
    
    #color balance
    cmin = np.percentile(rat2,0.1)
    cmax = np.percentile(rat2,99.9)
    rat2[np.where(rat2<cmin)] = cmin
    rat2[np.where(rat2>cmax)] = cmax
    #alpha composit
    
    rat2 = gf.to_8_bit(rat2)   
    rat2 = rat2[:,:,:,None]*np.ones(4,dtype = np.uint8)[None,:]
    
    wh = np.where(type_overlay[...,-1] > 0)
    rat2[:,wh[0],wh[1],:] = type_overlay[wh[0],wh[1],:]
    break
    tifffile.imsave(Path(viewing_dir,data.use, f'{data.trial_string}_overlay_type.tif'),gf.to_8_bit(rat2))

    '''
    rat = rat2[:,2:-2,2:-2]
    rat = ndimage.filters.gaussian_filter(rat,(3,2,2))
    rat = np.pad(rat,((0,0),(2,2),(2,2)),mode = 'edge')[::2,...]
    tifffile.imsave(Path(viewing_dir, f'{data.trial_string}_overlay_2.tif'),gf.to_8_bit(rat))
    '''
