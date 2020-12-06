#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:55:44 2020

@author: peter
"""

#a scratch process long im script

import numpy as np
from pathlib import Path
import pandas as pd

import cancer_functions as canf

    
    
def load_all_long(df_file,save_dir,redo = True, HPC_num = None):
    
    df = pd.read_csv(df_file)
    
    if redo == True:
        failed = []
        redo_from = 0
    else:
        redo_from = np.load(Path(save_dir,f'{df_file.stem}_redo_from.npy'))
        try:
            failed = list(pd.read_csv(Path(save_dir,'failed.csv')).index)
        except FileNotFoundError:
            failed = []
            
    
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
            if Path(trial_save,f'{trial_string}_ratio_stack.npy').is_file():
                continue
                
                
        try:
            result_dict = canf.load_and_slice_long_ratio(data.tif_file,
                                                         str(data.SMR_file),
                                                         T_approx = 3*10**-3,
                                                         fs = 5)
        except ValueError as err:
            if HPC_num is not None:
                raise err
            print(err)
            failed.append(data.Index)
            redo_from += 1
            df.loc[failed].to_csv( Path(save_dir,'failed.csv'))
            continue
            
    
        
        if not trial_save.is_dir():
            trial_save.mkdir(parents = True)
        
        for key in result_dict.keys():
            np.save(Path(trial_save,f'{trial_string}_{key}.npy'),result_dict[key])
        
        print(f'Saved {trial_string}')
        redo_from += 1
        np.save(Path(save_dir,f'{df_file.stem}_redo_from.npy'),redo_from)
    
    if failed != []:
        df.loc[failed].to_csv( Path(save_dir,'failed.csv'))
        
    not_failed = [i for i in df.index if i not in failed]
    
    return df.loc[not_failed],df.loc[failed]
    