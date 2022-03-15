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

import pdb

def export_events(initial_df,save_dir,thresh_idx,min_ttx_amp = 1, amp_threshold = None):

    df = pd.read_csv(initial_df)
    
    user_input_df = pd.read_csv(Path(save_dir,'good_detections.csv'))

    user_input_df.to_csv(Path(save_dir,'good_detections_debug_new.csv'))
    
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
            

            row = user_input_df[(user_input_df.trial_string == data.trial_string)&(user_input_df.cell_id == c_id)]
            if len(row)!= 1:
                raise ValueError('Should only be one matching cell')
            
            use = bool(row.correct.values[0])
            if not use:
                continue
            
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
        
        active_cell_use = [user_input_df[(user_input_df.trial_string == data.trial_string)&(user_input_df.cell_id == x)].correct.values[0] for x in active_cell_ids]
        active_cell_ids = list(np.array(active_cell_ids)[active_cell_use])

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
        
        tot_cells.append(events['tc'].shape[0] - np.sum(np.logical_not(active_cell_use))) # have to get rid of exclusions
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
    
    
    
    
    #now do per cell events
    
    
    all_cell_id = []
    num_pos_evs = []
    num_neg_evs = []
    all_integ_ev = []
    neg_integ_ev = []
    trial = []
    day = []
    slip = []
    expts = []
    stage = []
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
        active_cell_use = [user_input_df[(user_input_df.trial_string == data.trial_string)&(user_input_df.cell_id == x)].correct.values[0] for x in active_cell_ids]
       
        dont_use = list(np.array(active_cell_ids)[np.logical_not(active_cell_use)])
        active_cell_ids = list(np.array(active_cell_ids)[active_cell_use])

        
        all_cell_ids = [x for x in range(events['tc'].shape[0])]

        for c_id in all_cell_ids:
            
            cell_name = f'{trial_string}_cell_{c_id}'            
            
            if c_id in active_cell_ids:
                eve = events[c_id]
                eve_prop = events['event_props'][c_id]
                
                #remove too large events
                eve_prop = eve_prop[np.abs(eve_prop[:,1])<0.066,:]
                
                if amp_threshold is not None:
                    eve_prop = eve_prop[np.abs(eve_prop[:,1])>amp_threshold,:]
                
                n_pos_evs = np.sum(eve_prop[:,1] > 0)
                n_neg_evs = np.sum(eve_prop[:,1] < 0)
                sum_integ_evs = np.sum(np.abs(eve_prop[:,2]))
                sum_neg_integ_evs = np.sum(eve_prop[eve_prop[:,1] < 0,2])
            elif c_id in dont_use:
                continue
                
            else:
                n_pos_evs = 0
                n_neg_evs = 0
                sum_integ_evs = 0
                sum_neg_integ_evs = 0
                
            
            all_cell_id.append(cell_name)
            num_pos_evs.append(n_pos_evs)
            num_neg_evs.append(n_neg_evs)
            all_integ_ev.append(sum_integ_evs)
            neg_integ_ev.append(sum_neg_integ_evs)
            trial.append(data.trial_string)
            day.append(data.date)
            slip.append(data.slip)
            expts.append(data.expt)
            stage.append(data.stage)
        

            obs_length.append(results['observation_length'][thresh_idx][c_id])
    
        

    TTX_df2 = pd.DataFrame({'cell':all_cell_id,
                            'trial':trial,
                           'day':day,    
                            'n_pos_events': num_pos_evs,
                            'n_neg_events': num_neg_evs,
                            'integrated_events': all_integ_ev,
                            'neg_integrated_events': neg_integ_ev,
                           'slip':slip,
                           'expt':expts,
                           'stage':stage,
                           'obs_length':obs_length})


    TTX_df2.to_csv(Path(save_dir,'TTX_active_df_by_cell.csv'))
    
    
    #now do per cell events for non ttx
    
    
    all_cell_id = []
    num_pos_evs = []
    num_neg_evs = []
    all_integ_ev = []
    neg_integ_ev = []
    trial = []
    day = []
    slip = []
    expts = []
    stage = []
    obs_length = []
    
    for idx,data in enumerate(df.itertuples()):
        
        if 'TTX' in data.expt:
            continue
        
        if data.use != 'y':
            continue
        
        if 'washin' in data.expt:
            continue
        
        trial_string = data.trial_string
        #if trial_string == 'cancer_20210314_slip2_area3_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.06681_green_0.07975_1':
        #    pdb.set_trace()
        
        trial_save = Path(save_dir,'ratio_stacks',trial_string)

        results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
        events = results['events'][thresh_idx]
        

        
        active_cell_ids = [x for x in events.keys() if type(x) != str]
        active_cell_use = [bool(user_input_df[(user_input_df.trial_string == data.trial_string)&(user_input_df.cell_id == x)].correct.values[0]) for x in active_cell_ids]
      
        dont_use = list(np.array(active_cell_ids)[np.logical_not(active_cell_use)])
        active_cell_ids = list(np.array(active_cell_ids)[active_cell_use])
        

        
        
        all_cell_ids = [x for x in range(events['tc'].shape[0])]

        for c_id in all_cell_ids:            
            
            cell_name = f'{trial_string}_cell_{c_id}'            
            
            if c_id in active_cell_ids:
                eve = events[c_id]
                eve_prop = events['event_props'][c_id]
                
                #remove too large events
                eve_prop = eve_prop[np.abs(eve_prop[:,1])<0.066,:]
                
                if amp_threshold is not None:
                    eve_prop = eve_prop[np.abs(eve_prop[:,1])>amp_threshold,:]
                
                n_pos_evs = np.sum(eve_prop[:,1] > 0)
                n_neg_evs = np.sum(eve_prop[:,1] < 0)
                sum_integ_evs = np.sum(np.abs(eve_prop[:,2]))
                sum_neg_integ_evs = np.sum(eve_prop[eve_prop[:,1] < 0,2])
            elif c_id in dont_use:
                continue
                
            else:
                n_pos_evs = 0
                n_neg_evs = 0
                sum_integ_evs = 0
                sum_neg_integ_evs = 0
                
            
            all_cell_id.append(cell_name)
            num_pos_evs.append(n_pos_evs)
            num_neg_evs.append(n_neg_evs)
            all_integ_ev.append(sum_integ_evs)
            neg_integ_ev.append(sum_neg_integ_evs)
            trial.append(data.trial_string)
            day.append(data.date)
            slip.append(data.slip)
            expts.append(data.expt)
            if type(data.stage) == str:
                stage.append(data.stage)
            else:
                stage.append('none')
            

            obs_length.append(results['observation_length'][thresh_idx][c_id])

            
    
        

    df3 = pd.DataFrame({'cell':all_cell_id,
                            'trial':trial,
                           'day':day,    
                            'n_pos_events': num_pos_evs,
                            'n_neg_events': num_neg_evs,
                            'integrated_events': all_integ_ev,
                            'neg_integrated_events': neg_integ_ev,
                           'slip':slip,
                           'expt':expts,
                           'stage':stage,
                           'obs_length':obs_length})


    df3.to_csv(Path(save_dir,'non_ttx_active_df_by_cell.csv'))
        
        
        

        
        
        