#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:45:49 2020

@author: peter
"""

import numpy as np
from pathlib import Path
import pandas as pd
import tifffile

from joblib import Parallel, delayed
import cancer_functions as canf





def make_all_raw_tc(df_file,save_dir, redo = True,njobs = 10, HPC_num = None):
    df = pd.read_csv(df_file)
    
    if redo:
        redo_from = 0
    else:
        redo_from = np.load(Path(save_dir,f'{df_file.stem}_redo_from_make_all_raw_tc.npy'))
        print(f'{len(df) - redo_from} to do')

    

    for idx,data in enumerate(df.itertuples()):
        if HPC_num is not None: #allows running in parallel on HPC
            if idx != HPC_num:
                continue
        
        parts = Path(data.tif_file).parts
        trial_string = '_'.join(parts[parts.index('cancer'):-1])
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        
        
        if not redo and HPC_num is None:
            if idx < redo_from:
                continue
        elif not redo and HPC_num is not None:
            if Path(trial_save,f'{trial_string}_all_cellfree_tc.npy').is_file():
                continue
    
        seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
        masks = canf.lab2masks(seg)
        
        LED = np.load(Path(trial_save,f'{trial_string}_LED_powers.npy'))
        if LED[0] < LED[1]:
            blue = 0
        else:
            blue = 1
        
        stack = tifffile.imread(data.tif_file)[blue::2,...]
        

        if HPC_num is None:
            with Parallel(n_jobs=njobs) as parallel:
                tc = parallel(delayed(canf.t_course_from_roi)(stack,mask) for mask in masks)

        else:
            tc = [canf.t_course_from_roi(stack,mask) for mask in masks]

    
        tc = np.array(tc)

        

        
        np.save(Path(trial_save,f'{trial_string}_raw_tc.npy'),tc)


        print(f'Saved {trial_string}')
        redo_from += 1
        np.save(Path(save_dir,f'{df_file.stem}_redo_from_make_all_raw_tc.npy'),redo_from)
        