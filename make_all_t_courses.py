#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:45:49 2020

@author: peter
"""

import numpy as np
from pathlib import Path
import pandas as pd

from joblib import Parallel, delayed


def t_course_from_roi(nd_stack,roi):
    masked = np.ma.masked_less(roi.astype(int),1)
    if len(roi.shape) == 2:
        return np.mean(np.mean(nd_stack*masked[None,...],-1),-1).data
    else:
        sh1 = nd_stack.shape #assume [...,t,y,x]
        sh2 = roi.shape #assume [...,y,x] and want to keep all dims
        i = len(sh1) - 2
        j = len(sh2) - 2
        nd_stack = nd_stack.reshape(sh1[:-3] + tuple(np.ones(j,dtype = int))+sh1[-3:])
        masked = masked.reshape(sh2[:-2]+tuple(np.ones(i,dtype = int))+sh2[-2:])
        return np.mean(np.mean(nd_stack*masked,-1),-1).data

def lab2masks(seg):
    masks = []
    for i in range(1,seg.max()):
        masks.append((seg == i).astype(int))
    return np.array(masks)


def make_all_tc(df_file,save_dir, redo = True, njobs = 2, HPC_num = None):
    df = pd.read_csv(df_file)
    
    if HPC_num is not None:
        njobs = 1
    
    if redo:
        redo_from = 0
    else:
        redo_from = np.load(Path(save_dir,f'{df_file.stem}_redo_from_make_all_tc.npy'))
        print(f'{len(df) - redo_from} to do')

    
    with Parallel(n_jobs=njobs) as parallel:
        for idx,data in enumerate(df.itertuples()):
            if HPC_num is not None: #allows running in parallel on HPC
                if idx != HPC_num:
                    continue
            
            if idx < redo_from:
                continue
            
            parts = Path(data.tif_file).parts
            trial_string = '_'.join(parts[parts.index('cancer'):-1])
            trial_save = Path(save_dir,'ratio_stacks',trial_string)
            
            seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
            
            masks = lab2masks(seg)
            
            stack = np.load(Path(trial_save,f'{trial_string}_ratio_stack.npy'))
            
            try:
                tc = parallel(delayed(t_course_from_roi)(stack,mask) for mask in masks)
            except Exception as err:
                print(err)
                tc = [t_course_from_roi(stack,mask) for mask in masks]
        
            tc = np.array(tc)
            
            np.save(Path(trial_save,f'{trial_string}_all_tcs.npy'),tc)
            
            print(f'Saved {trial_string}')
            redo_from += 1
            np.save(Path(save_dir,f'{df_file.stem}_redo_from_make_all_tc.npy'),redo_from)
        