#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 13:01:46 2020

@author: peter
"""


import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import ast

from cellpose import models
import ruptures as rpt

import cancer_functions as canf



def main(num,df_path,redo_load = True,redo_tc = True):
    top_dir = Path(Path.home(),'firefly_link/cancer')
        
    save_dir = Path(top_dir,'analysis','full')
    
    df = pd.read_csv(df_path)
    
    for idx,data in enumerate(df.itertuples()):
        if idx != num:
            continue
        
        trial_string = data.trial_string
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        if not trial_save.is_dir():
            trial_save.mkdir(parents = True)
                
        if 'washin' in data.expt:
            washin = True #want to use a causal filter 
        else:
            washin = False
            
            
        ################# Make ratio ################
        
        def local_to_HPC(tif_path):
            if '/home/peter/data/' in str(tif_path):
                new_path = Path(Path.home(),'firefly_link',*Path(tif_path).parts[5:])
            return new_path
        
        if Path(trial_save,f'{trial_string}_ratio_stack.npy').is_file() and not redo_load:
            print('Loading stack')
            keys = ['ratio_stack','im']
            result_dict = {}
            for key in keys:
                if key == 'ratio_stack':
                    result_dict[key] = np.load(Path(trial_save,f'{trial_string}_{key}.npy')).astype(np.float64)
                else:
                    result_dict[key] = np.load(Path(trial_save,f'{trial_string}_{key}.npy'))
        else:
            print('Calculating stack')
            result_dict = canf.load_and_slice_long_ratio(local_to_HPC(data.tif_file),
                                                             str(local_to_HPC(data.SMR_file)),
                                                             T_approx = 3*10**-3,
                                                             fs = 5,
                                                             washin = washin)
        
            for key in result_dict.keys():
                if key == 'ratio_stack':
                    np.save(Path(trial_save,f'{trial_string}_{key}.npy'),result_dict[key].astype(np.float32))
                else:
                    np.save(Path(trial_save,f'{trial_string}_{key}.npy'),result_dict[key])
                
        
        ########### Segment ###################
        print('Segmenting')
        
        model = models.Cellpose(gpu=False, model_type='cyto')
        masks, flows, styles, diams = model.eval(result_dict['im'], diameter=30, channels=[0,0])
        np.save(Path(trial_save,f'{trial_string}_seg.npy'),np.squeeze(masks))
    
    
        ########### Extract TC #################
        print('Extracting time courses')
        
        def t_course_from_roi(stack,roi):
            masked = np.ma.masked_less(roi.astype(int),1)
            return np.mean(np.mean(stack*masked[None,...],-1),-1).data
        
        def lab2masks(seg):
            masks = []
            for i in range(1,seg.max()):
                masks.append((seg == i).astype(int))
            return np.array(masks)
        
        masks = lab2masks(np.squeeze(masks))
    
        if Path(trial_save,f'{trial_string}_all_tcs.npy').is_file() and not redo_tc:
            print('Calculating TCs')
            tc = np.array([t_course_from_roi(result_dict['ratio_stack'],mask) for mask in masks])
            np.save(Path(trial_save,f'{trial_string}_all_tcs.npy'),tc)
        else:
            print('Loading TCs')
            tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
            
            
        ########### Detect events ###############
        print('Detecting Events')
        
        
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
    
        ev = [get_change_points(t) for t in tc]
        
        ev = np.array(ev,dtype = object)
    
        np.save(Path(trial_save,f'{trial_string}_detected_events.npy'),ev)

        print('Done!')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num', type = float, help = 'PBS array job number')
    parser.add_argument('df_path', type = str, help = 'Path to dataframe')
    
    parser.add_argument('--redo_load',dest = 'redo_load' ,type = str,  default ='True')
    parser.add_argument('--redo_tc',dest = 'redo_tc' ,type = str,  default ='True')

    args = parser.parse_args()
    
    args.redo_load = ast.literal_eval(args.redo_load)
    args.redo_tc = ast.literal_eval(args.redo_tc)

    args.num = int(args.num) # for failed nums
    args.num = args.num - 1 #PBS uses 1 based indexing, convert to 0 base

    for key in vars(args):
        print(f'{key}: {vars(args)[key]}')

    main(args.num, 
        args.df_path, 
        redo_load = args.redo_load,
        redo_tc = args.redo_tc)
        