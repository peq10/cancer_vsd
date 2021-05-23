#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:14:32 2020

@author: peter
"""
import numpy as np
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

import ruptures as rpt

def get_change_points(t, model_params):
    if model_params is None:
        model = "rank"  # "l2", "rbf"
        min_size = 3
        jump = 5
        penalty = 20
    else:
        model = model_params['model']
        min_size = model_params['min_size']
        jump = model_params['jump']
        penalty = model_params['penalty']
        
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
    result = algo.fit_predict(t,penalty)
    
    return result


def detect_all_events(df_file,save_dir, redo = True, njobs = 2, debug = False, model_params = None, HPC_num = None):
    df = pd.read_csv(df_file)
        
    if redo:
        redo_from = 0
    else:
        redo_from = np.load(Path(save_dir,f'{df_file.stem}_redo_from_detect_events.npy'))
    
    print(f'{len(df) - redo_from} to do')
    
    
    for idx,data in enumerate(df.itertuples()):
        
        if HPC_num is not None: #allows running in parallel on HPC
            if idx != HPC_num:
                continue
    
        if '20201215_slip2_area1_long_acq_corr' in data.trial_string:
            break
    
        parts = Path(data.tif_file).parts
        trial_string = '_'.join(parts[parts.index('cancer'):-1])
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        if not redo and HPC_num is None:
            if idx < redo_from:
                continue
        elif not redo and HPC_num is not None:
            if Path(trial_save,f'{trial_string}_detected_events.npy').is_file():
                continue    

        tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'),allow_pickle = True)
        median_tc = np.load(Path(trial_save,f'{trial_string}_all_median_tcs.npy'),allow_pickle = True)
        
        eroded_tc = np.load(Path(trial_save,f'{trial_string}_all_eroded_tcs.npy'),allow_pickle = True)
        eroded_median_tc = np.load(Path(trial_save,f'{trial_string}_all_eroded_median_tcs.npy'),allow_pickle = True)
        
        if debug:
            tc = tc[:10,:]
        
        if HPC_num is None:
            try:
                with Parallel(n_jobs=njobs) as parallel:
                    ev = parallel(delayed(get_change_points)(t,model_params) for t in tc)
            except Exception as err:
                print(err)
                ev = [get_change_points(t) for t in tc]
        else:
            ev = [get_change_points(t) for t in tc]
            
        ev = np.array(ev,dtype = object)
    
        np.save(Path(trial_save,f'{trial_string}_detected_events.npy'),ev)

        print(f'Saved {trial_string}')
        redo_from += 1
        np.save(Path(save_dir,f'{df_file.stem}_redo_from_detect_all_event.npy'),redo_from)
            
            