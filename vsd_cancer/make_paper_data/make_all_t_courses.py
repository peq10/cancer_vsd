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

import scipy.ndimage as ndimage

from vsd_cancer.functions import cancer_functions as canf




def make_all_tc(df_file,save_dir, redo = True, njobs = 2, HPC_num = None,only_hand_rois = False):
    df = pd.read_csv(df_file)
    
    if redo:
        redo_from = 0
    else:
        redo_from = np.load(Path(save_dir,f'{df_file.stem}_redo_from_make_all_tc.npy'))
        print(f'{len(df) - redo_from} to do')

    

    for idx,data in enumerate(df.itertuples()):
        if HPC_num is not None: #allows running in parallel on HPC
            if idx != HPC_num:
                continue
        
        parts = Path(data.tif_file).parts
        trial_string = '_'.join(parts[parts.index('cancer'):-1])
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        
        if only_hand_rois: #just faster when doing hand rois
            if not Path(trial_save,'hand_rois').is_dir():
                continue
            
        if not redo and HPC_num is None:
            if idx < redo_from:
                continue
        elif not redo and HPC_num is not None:
            if Path(trial_save,f'{trial_string}_all_tcs.npy').is_file():
                continue
    
        seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
        
        masks = canf.lab2masks(seg)
        surround_masks = canf.get_surround_masks_cellfree(masks, dilate = True)
        
        structure = np.zeros((3,3,3))
        structure[1,:,:] = 1
        eroded_masks = ndimage.binary_erosion(masks,structure)
        #also make eroded masks to avoid movement artefacts around outside
        
        
        stack = np.load(Path(trial_save,f'{trial_string}_ratio_stack.npy')).astype(np.float64)
        
        if HPC_num is None:
            try:
                with Parallel(n_jobs=njobs) as parallel:
                    tc = parallel(delayed(canf.t_course_from_roi)(stack,mask) for mask in masks)
                    eroded_tc = parallel(delayed(canf.t_course_from_roi)(stack,mask) for mask in eroded_masks)
                    median_tc = parallel(delayed(canf.median_t_course_from_roi)(stack,mask) for mask in masks)
                    eroded_median_tc = parallel(delayed(canf.median_t_course_from_roi)(stack,mask) for mask in eroded_masks)
                    surround_tc = parallel(delayed(canf.t_course_from_roi)(stack,mask) for mask in surround_masks)
                    median_surround_tc = parallel(delayed(canf.median_t_course_from_roi)(stack,mask) for mask in surround_masks)
                    std = parallel(delayed(canf.std_t_course_from_roi)(stack, mask,True) for mask in masks)
                    surround_std = parallel(delayed(canf.std_t_course_from_roi)(stack, mask, True) for mask in masks)
            except Exception as err:
                print(err)
                tc = [canf.t_course_from_roi(stack,mask) for mask in masks]
                eroded_tc = [canf.t_course_from_roi(stack,mask) for mask in eroded_masks]
                median_tc = [canf.median_t_course_from_roi(stack,mask) for mask in masks]
                eroded_median_tc = [canf.median_t_course_from_roi(stack,mask) for mask in eroded_masks]
                surround_tc = [canf.t_course_from_roi(stack,mask) for mask in surround_masks]
                median_surround_tc = [canf.median_t_course_from_roi(stack,mask) for mask in surround_masks]
                std = [canf.std_t_course_from_roi(stack, mask, True) for mask in masks]
                surround_std = [canf.std_t_course_from_roi(stack, mask,True) for mask in masks]
        else:
            tc = [canf.t_course_from_roi(stack,mask) for mask in masks]
            eroded_tc = [canf.t_course_from_roi(stack,mask) for mask in eroded_masks]
            median_tc = [canf.median_t_course_from_roi(stack,mask) for mask in masks]
            eroded_median_tc = [canf.median_t_course_from_roi(stack,mask) for mask in eroded_masks]
            surround_tc = [canf.t_course_from_roi(stack,mask) for mask in surround_masks]
            median_surround_tc = [canf.median_t_course_from_roi(stack,mask) for mask in surround_masks]
            std = [canf.std_t_course_from_roi(stack, mask, True) for mask in masks]
            surround_std = [canf.std_t_course_from_roi(stack, mask, True) for mask in masks]
    
        tc = np.array(tc)
        tc -= tc.mean(-1)[:,None] - 1
        
        eroded_tc = np.array(eroded_tc)
        eroded_tc -= eroded_tc.mean(-1)[:,None] - 1
        
        median_tc = np.array(median_tc)
        median_tc -= median_tc.mean(-1)[:,None] - 1
        
        eroded_median_tc = np.array(eroded_median_tc)
        eroded_median_tc -= eroded_median_tc.mean(-1)[:,None] - 1
        
        surround_tc = np.array(surround_tc)
        surround_tc -= surround_tc.mean(-1)[:,None] - 1
        
        median_surround_tc = np.array(median_surround_tc)
        median_surround_tc -= median_surround_tc.mean(-1)[:,None] - 1
        
        
        np.save(Path(trial_save,f'{trial_string}_all_tcs.npy'),tc)
        np.save(Path(trial_save,f'{trial_string}_all_eroded_tcs.npy'),eroded_tc)
        np.save(Path(trial_save,f'{trial_string}_all_median_tcs.npy'),median_tc)
        np.save(Path(trial_save,f'{trial_string}_all_eroded_median_tcs.npy'),eroded_median_tc)
        np.save(Path(trial_save,f'{trial_string}_all_surround_tcs.npy'),surround_tc)
        np.save(Path(trial_save,f'{trial_string}_all_median_surround_tcs.npy'),median_surround_tc)
        np.save(Path(trial_save,f'{trial_string}_all_stds.npy'),std)
        np.save(Path(trial_save,f'{trial_string}_all_surround_stds.npy'),surround_std)
        
        print(f'Saved {trial_string}')
        redo_from += 1
        if not only_hand_rois:
            np.save(Path(save_dir,f'{df_file.stem}_redo_from_make_all_tc.npy'),redo_from)
        
        