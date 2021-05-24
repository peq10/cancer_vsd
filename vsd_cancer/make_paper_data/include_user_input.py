#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 19:40:30 2021

@author: peter
"""

import pandas as pd
import numpy as np
from pathlib import Path

import copy


def include_user_input(initial_df,save_dir,thresh_idx):
    df = pd.read_csv(initial_df)
    df = df[(df.use == 'y') & ((df.expt == 'MCF10A')|(df.expt == 'MCF10A_TGFB'))]
    print('ONLY DOING 10As')
    
    trial_string = df.iloc[0].trial_string
    
    for idx,data in enumerate(df.itertuples()):
        trial_string = data.trial_string
        #print(trial_string)
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
    
        if data.use == 'n':
            continue
        
        
   
        
        new_results = {key:0 for key in results.keys()}
        
        new_results['excluded_circle'] = copy.deepcopy(results['excluded_circle'])
        new_results['detection_params'] = copy.deepcopy(results['detection_params'])
        new_results['observation_length'] = copy.deepcopy(results['observation_length'][thresh_idx])
        
        new_events = results['events'][thresh_idx]
        new_events['user_excluded_events'] = {}
        new_events['user_excluded_event_props'] = {}
        
        
        #iterate through and check if excluded by user input
        to_delete = []
        for key in new_events.keys():
            if type(key) == str:
                continue
        
            user_input = np.load(Path(trial_save,f'{trial_string}_good_detection_cell_{key}.npy'))
            if user_input == False:
                to_delete.append(key)
            
        
        for cell in to_delete:
            new_events['user_excluded_events'][cell] = copy.deepcopy(new_events[cell])
            del new_events[cell]
            #also move event properties
            new_events['user_excluded_event_props'][cell] = copy.deepcopy(new_events['event_props'][cell])
            del new_events['event_props'][cell]
            #also need to modify observation length
            new_results['observation_length'][cell] = 0# if user excludes, its for whole video
        
        
        new_results['events'] = new_events
        
        
        np.save(Path(trial_save,f'{trial_string}_event_properties_including_user_input.npy'),new_results)
