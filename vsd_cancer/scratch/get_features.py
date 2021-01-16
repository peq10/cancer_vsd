#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:04:30 2021

@author: peter
"""

import numpy as np

import pandas as pd
from pathlib import Path

import cancer_functions as canf

import catch22


def get_measure_events(initial_df,save_dir,thresh = 0.002,
                       surrounds_thresh = 0.001,
                       exclude_first = 200):

    df = pd.read_csv(initial_df)
    for idx, data in enumerate(df.itertuples()):
        #if idx != 34:
        #    continue
        
        trial_string = data.trial_string
        print(trial_string)
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    
        filt_params = {'type':'gaussian','gaussian_sigma':3}
        
        excluded_circle = np.load(Path(trial_save,f'{trial_string}_circle_excluded_rois.npy'))
        #also get circle exclusions
        surround_tc = np.load(Path(trial_save,f'{trial_string}_all_surround_tcs.npy'))
        
        if not np.isnan(data.finish_at):
            observe_to = int(data.finish_at)*5
            tc = tc[:,:observe_to]
            surround_tc = surround_tc[:observe_to]
        


        events = canf.get_events_exclude_surround_events(tc,
                                                         surround_tc,
                                                         detection_thresh = thresh, 
                                                         surrounds_thresh = surrounds_thresh,
                                                         filt_params = filt_params, 
                                                         exclude_first=exclude_first, 
                                                         excluded_circle = excluded_circle)
        
        active = np.array([x for x in events.keys() if type(x) != str])
        results = np.zeros((len(active),23))
        results[:,0] = active
        
        tcs = events['tc_filt']
        
        result_dict = {'tc_filt':tcs,'cells':active}
        
        for idx2,tc in enumerate(tcs[active,...]):
            feat = catch22.catch22_all(tc)
            results[idx2,1:] = feat['values']
            
        result_dict['features'] = results
        np.save(Path(trial_save,f'{trial_string}_event_properties.npy'),result_dict)