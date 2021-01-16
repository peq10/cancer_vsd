#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:09:07 2021

@author: peter
"""
#a script to get all the events

import numpy as np

import pandas as pd
from pathlib import Path

import cancer_functions as canf




def get_measure_events(initial_df,save_dir,thresh_range = np.arange(2,4.5,0.5),
                       surrounds_z = 7,
                       exclude_first = 0):

    df = pd.read_csv(initial_df)
    for idx, data in enumerate(df.itertuples()):
        #if idx != 34:
        #    continue
        
        trial_string = data.trial_string
        print(trial_string)
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
        tc -= np.mean(tc,-1)[:,None] - 1

        std = np.load(Path(trial_save,f'{trial_string}_all_stds.npy'))
        surround_std = np.load(Path(trial_save,f'{trial_string}_all_surround_stds.npy'))
        
        excluded_circle = np.load(Path(trial_save,f'{trial_string}_circle_excluded_rois.npy'))
        #also get circle exclusions
        surround_tc = np.load(Path(trial_save,f'{trial_string}_all_surround_tcs.npy'))
        #remove any surround offsets 
        surround_tc -= np.mean(surround_tc,-1)[:,None] - 1
        
        
        if not np.isnan(data.finish_at):
            observe_to = int(data.finish_at)*5
            tc = tc[:,:observe_to]
            std = std[:,:observe_to]
            surround_tc = surround_tc[:observe_to]
            surround_std = surround_std[:,:observe_to]
        
        all_events = []
        all_observation = []
        for detection_thresh in thresh_range:
            
            events = canf.get_events_exclude_surround_events(tc,
                                                             std,
                                                             surround_tc,
                                                             surround_std,
                                                             z_score = detection_thresh, 
                                                             surround_z = surrounds_z,
                                                             exclude_first= exclude_first, 
                                                             excluded_circle = None)
        

            event_with_props = canf.get_event_properties(events,use_filt = False) 
            
            all_events.append(event_with_props)
            all_observation.append(canf.get_observation_length(events))
            
        
        
        detect_params = {'thresh_range':thresh_range,
                         'surrounds_thresh':surrounds_z,
                         'exclude_first':exclude_first}
        
        result_dict = {'n_cells': tc.shape[0] - len(excluded_circle),
                      'events': all_events,
                      'observation_length': all_observation,
                      'excluded_circle': excluded_circle,
                      'detection_params': detect_params
                      }
        
        #all_props = np.concatenate([event_props[p] for p in event_props.keys() if 'props' in str(p)])
        
        np.save(Path(trial_save,f'{trial_string}_event_properties.npy'),result_dict)
        