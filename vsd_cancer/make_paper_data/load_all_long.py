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

from vsd_cancer.functions import cancer_functions as canf

    
    
def load_all_long(df_file,save_dir,redo = True, HPC_num = None, raise_err = False):
    
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
              
        
        trial_string = data.trial_string
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        print(trial_string)
        print('hello')
        if not redo and HPC_num is None:
            if idx < redo_from:
                continue
        elif not redo and HPC_num is not None:
            if Path(trial_save,f'{trial_string}_ratio_stack.npy').is_file():
                continue
                
        if 'washin' in data.expt:
            washin = True #want to use a causal filter 
        else:
            washin = False
        
        
        try:
            result_dict = canf.load_and_slice_long_ratio(data.tif_file,
                                                         str(data.SMR_file),
                                                         T_approx = 3*10**-3,
                                                         fs = 5,
                                                         washin = washin)
        except ValueError as err:
            
            if raise_err:
                raise err
            else:
                if HPC_num is not None:
                    raise err
                print(err)
                failed.append(data.Index)
                redo_from += 1
                fail_df = Path(save_dir,'failed.csv')
                df.loc[failed].to_csv(fail_df,mode = 'a',header = not fail_df.is_file())
                
                continue
            
    
        
        if not trial_save.is_dir():
            trial_save.mkdir(parents = True)
        
        for key in result_dict.keys():
            if key == 'ratio_stack':
                np.save(Path(trial_save,f'{trial_string}_{key}.npy'),result_dict[key].astype(np.float32))
            else:
                np.save(Path(trial_save,f'{trial_string}_{key}.npy'),result_dict[key])
        
        print(f'Saved {trial_string}')
        redo_from += 1
        np.save(Path(save_dir,f'{df_file.stem}_redo_from.npy'),redo_from)
    
    if failed != []:
        df.loc[failed].to_csv( Path(save_dir,'failed.csv'))
        
    not_failed = [i for i in df.index if i not in failed]
    
    return df.loc[not_failed],df.loc[failed]

def detect_failed(df_file,save_dir):
    df = pd.read_csv(df_file)
    failed_df = pd.DataFrame()
    for idx,data in enumerate(df.itertuples()):
        trial_string = data.trial_string
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        if not Path(trial_save,f'{trial_string}_ratio_stack.npy').is_file():
            failed_df = failed_df.append(df.loc[data.Index])
    failed_df.to_csv(Path(save_dir,'actual_failed.csv'))
    return failed_df
    
def load_failed(failed_df_file,save_dir):
    
    df = pd.read_csv(failed_df_file)

    for idx,data in enumerate(df.itertuples()):

        
        trial_string = data.trial_string
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        print(trial_string)
        

                
        if 'washin' in data.expt:
            washin = True #want to use a causal filter 
        else:
            washin = False
        
        import pdb
        #pdb.set_trace()
        result_dict = canf.load_and_slice_long_ratio(data.tif_file,
                                                     str(data.SMR_file),
                                                     T_approx = 3*10**-3,
                                                     fs = 5,
                                                     washin = washin)

            
    
        
        if not trial_save.is_dir():
            trial_save.mkdir(parents = True)
        
        for key in result_dict.keys():
            if key == 'ratio_stack':
                np.save(Path(trial_save,f'{trial_string}_{key}.npy'),result_dict[key].astype(np.float32))
            else:
                np.save(Path(trial_save,f'{trial_string}_{key}.npy'),result_dict[key])
        
        print(f'Saved {trial_string}')

def load_all_long_washin(df_file,save_dir,redo = True, HPC_num = None, raise_err = False):
    
    df = pd.read_csv(df_file)
    
    df = df[(df.use == 'y') & ((df.expt == 'high_k_washin')|(df.expt == 'TTX_washin'))]
    
    if redo == True:
        failed = []
        redo_from = 0
    else:
        redo_from = np.load(Path(save_dir,f'{df_file.stem}_redo_from_washin.npy'))
        try:
            failed = list(pd.read_csv(Path(save_dir,'failed_washin.csv')).index)
        except FileNotFoundError:
            failed = []
            
    
    for idx,data in enumerate(df.itertuples()):
        if HPC_num is not None: #allows running in parallel on HPC
            if idx != HPC_num:
                continue
              
        
        trial_string = data.trial_string
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        print(trial_string)
        
        if not redo and HPC_num is None:
            if idx < redo_from:
                continue
        elif not redo and HPC_num is not None:
            if Path(trial_save,f'{trial_string}_ratio_stack_washin.npy').is_file():
                continue
                
        if 'washin' in data.expt:
            washin = True #want to use a causal filter 
        else:
            washin = False
            
        try:
            result_dict = canf.load_and_slice_long_ratio(data.tif_file,
                                                         str(data.SMR_file),
                                                         T_approx = 3*10**-3,
                                                         fs = 5,
                                                         washin = washin,
                                                         nofilt = True)
        except ValueError as err:
            
            if raise_err:
                raise err
            else:
                if HPC_num is not None:
                    raise err
                print(err)
                failed.append(data.Index)
                redo_from += 1
                fail_df = Path(save_dir,'failed_washin.csv')
                df.loc[failed].to_csv(fail_df,mode = 'a',header = not fail_df.is_file())
                
                continue
            
    
        
        if not trial_save.is_dir():
            trial_save.mkdir(parents = True)
        
        for key in result_dict.keys():
            if key == 'ratio_stack':
                np.save(Path(trial_save,f'{trial_string}_{key}_washin.npy'),result_dict[key].astype(np.float32))
            else:
                continue
        
        print(f'Saved {trial_string}')
        redo_from += 1
        np.save(Path(save_dir,f'{df_file.stem}_redo_from_washin.npy'),redo_from)
    
    if failed != []:
        df.loc[failed].to_csv( Path(save_dir,'failed_washin.csv'))
        
    not_failed = [i for i in df.index if i not in failed]
    
    return df.loc[not_failed],df.loc[failed]