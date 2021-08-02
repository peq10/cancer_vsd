#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 09:11:24 2021

@author: peter
"""
import numpy as np

from pathlib import Path

import pandas as pd

from vsd_cancer.functions import cancer_functions as canf

import scipy.ndimage as ndimage



def export_events(initial_df,save_dir,thresh_idx,min_ttx_amp = 1):

    df = pd.read_csv(initial_df)
    
    print('NEED TO CHANGE TO INCLUDE QC')
    
    all_cell_id = []
    all_cell_x = []
    all_cell_y = []
    all_event_time = []
    all_event_amplitude = []
    all_event_length = []
    all_event_integral = []
    all_expt = []
    all_std = []
    all_stage = []
    all_trial_str = []
    for idx,data in enumerate(df.itertuples()):
        
        if data.use == 'n':
            continue
        
        if 'washin' in data.expt:
            continue
        
        trial_string = data.trial_string
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
        seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
        masks = canf.lab2masks(seg)
    
        events = results['events'][thresh_idx]
        
        cell_ids = [x for x in events.keys() if type(x) != str]
        
        
        
        for c_id in cell_ids:
            
            cell_name = f'{trial_string}_cell_{c_id}'
            cell_y,cell_x = ndimage.measurements.center_of_mass(masks[c_id,...])
            std = np.std(events['tc'][c_id])
            
            eve = events[c_id]
            eve_prop = events['event_props'][c_id]
            
            for event_idx in range(eve.shape[-1]):
                event_time = np.mean(eve[:,event_idx])
                leng,amp,integ = eve_prop[event_idx,:]

                
                #add to df
                all_cell_id.append(cell_name)
                all_cell_x.append(cell_x)
                all_cell_y.append(cell_y)
                all_std.append(std)
                all_event_time.append(event_time)
                all_event_amplitude.append(amp)
                all_event_length.append(leng)
                all_event_integral.append(integ)
                all_expt.append(data.expt)
                all_trial_str.append(data.trial_string)
                
                if type(data.stage) == str:
                    all_stage.append(data.stage)
                else:
                    all_stage.append('none')
                    
    
    
    event_df = pd.DataFrame({'cell_id':all_cell_id,
                                    'event_time':all_event_time,
                                    'event_length':all_event_length,
                                    'event_amplitude':all_event_amplitude,
                                    'event_integrated':all_event_integral,
                                    'cell_x':all_cell_x,
                                    'cell_y':all_cell_y,
                                    'std':all_std,
                                    'expt':all_expt,
                                    'stage': all_stage,
                                    'trial_string': all_trial_str})
    
    
    event_df.to_csv(Path(save_dir,'all_events_df.csv'))
    
    
    #now get the number of active cells for TTX
    
    all_only_pos_active = []
    all_neg_active = []
    all_tot_active = []
    tot_cells = []
    trial = []
    day = []
    slip = []
    expts = []
    stage = []
    tot_cells = []
    tot_time = []
    obs_length = []
    
    for idx,data in enumerate(df.itertuples()):
        if 'TTX' not in data.expt:
            continue
        
        if data.use == 'n':
            continue
        
        if 'washin' in data.expt:
            continue
        
        trial_string = data.trial_string
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
        events = results['events'][thresh_idx]
        
        active_cell_ids = [x for x in events.keys() if type(x) != str]
        
        #exclude under minimum amplitude
        active_cell_ids = [x for x in active_cell_ids if np.max(np.abs(events['event_props'][x][:,1]))>min_ttx_amp/100]
        
        
        only_pos = [x for x in active_cell_ids if np.all(events['event_props'][x][:,1]>0)]
        
        num_active = len(active_cell_ids)
        only_pos_active = len(only_pos)
        neg_active = num_active - only_pos_active
        
        all_only_pos_active.append(only_pos_active)
        all_neg_active.append(neg_active)
        all_tot_active.append(num_active)
        
        trial.append(trial_string)
        day.append(data.date)
        slip.append(data.slip)
        expts.append(data.expt)
        stage.append(data.stage)
        
        tot_cells.append(events['tc'].shape[0])
        tot_time.append(events['tc'].shape[1])

        obs_length.append(np.sum(results['observation_length'][thresh_idx]))
        

    TTX_df = pd.DataFrame({'trial':trial,
                           'day':day,    
                           'pos_active':all_only_pos_active,
                           'neg_active':all_neg_active,
                           'total_active':all_tot_active,
                           'slip':slip,
                           'expt':expts,
                           'stage':stage,
                           'num_cells':tot_cells,
                           'imaging_length':tot_time,
                           'obs_length':obs_length})


    TTX_df.to_csv(Path(save_dir,'TTX_active_df.csv'))
        
        
        

        
        
        