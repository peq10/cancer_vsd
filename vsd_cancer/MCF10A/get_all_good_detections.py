#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:31:26 2021

@author: peter
"""

import numpy as np

import pandas as pd
from pathlib import Path

import scipy.ndimage as ndimage

import tifffile

import cv2

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)
df = df[(df.use == 'y') & ((df.expt == 'MCF10A')|(df.expt == 'MCF10A_TGFB'))]

trial_string = df.iloc[0].trial_string
n_thresh = len(np.load(Path(Path(save_dir,'ratio_stacks',trial_string),f'{trial_string}_event_properties.npy'),allow_pickle = True).item()['events'])

currents  = [[[],[]] for x in range(n_thresh)]
lengths  = [[[],[]] for x in range(n_thresh)]

detected_frame = pd.DataFrame()
detections = 0
use_idx = 1

for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    #print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
    seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
    cell_ids = np.arange(results['events'][0]['tc_filt'].shape[0])
    cell_ids = [x for x in cell_ids if x not in results['excluded_circle']]

    if data.use == 'n':
        continue
    
    
    for idx,thresh_level_dict in enumerate(results['events']):
        
        if idx != use_idx:
            continue
        
        event_props = results['events'][idx]['event_props']
        
        observations = [results['observation_length'][idx][x] for x in cell_ids]
        sum_current = [np.sum(np.abs(event_props[x][:,-1])) if x in event_props.keys() else 0 for x in cell_ids]
        
        #manually check finds
        if idx == use_idx:
            if np.any(np.array(sum_current)!=0):
                vidpath = [x for x in Path('/media/peter/bigdata/Firefly/cancer/analysis/full/tif_viewing/grey_videos/').glob(f'./**/*{trial_string}*')][0]
                vid = tifffile.imread(vidpath)
                
                active_cells = [x for x in results['events'][idx] if type(x)!= str]
                locs = np.round([ndimage.center_of_mass(seg == x+1) for x in active_cells]).astype(int)
                times = [results['events'][idx][x] for x in active_cells]
                for idxxx,ce in enumerate(active_cells):
                    detected_frame.loc[detections,'trial_string'] = trial_string
                    detected_frame.loc[detections,'cell_id'] = ce
                    detected_frame.loc[detections,'loc'] = str(locs[idxxx])
                    detected_frame.loc[detections,'starts'] = str(times[idxxx][0,:]/2)
                    detections+=1
                    #also make a small video around cell
                    if Path(trial_save,f'{trial_string}_good_detection_cell_{ce}.npy').is_file() and False:
                        detection_real = np.load(Path(trial_save,f'{trial_string}_good_detection_cell_{ce}.npy'))
                    else:
                        #raise ValueError('Have to do ONE PER DETECTION')
                        event_vid = vid[max(times[idxxx][0,0]//2-20,0):times[idxxx][1,-1]//2+20,max(locs[idxxx][0]-100,0):locs[idxxx][0]+100,max(locs[idxxx][1]-100,0):locs[idxxx][1]+100]
                        #label events with red spot in top left
                        for evv in times[idxxx].T:
                            t0 = times[idxxx][0,0]
                            event_vid[evv[0]-t0:evv[1]-t0,:10,:10] = 0
                        
                        ii = 0
                        windowname = f'{trial_string} Cell {ce}'
                        view_window = cv2.namedWindow(windowname,cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(windowname,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                        cv2.setWindowProperty(windowname,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(windowname,2000,2000)
                        while True:
                            
                            # Display the resulting frame
                            
                            cv2.imshow(windowname, event_vid[ii%event_vid.shape[0]])
                            
                            # Press Q on keyboard to  exit
                            if cv2.waitKey(10) & 0xFF == ord('y'):  
                                detection_real = True
                                break
                            elif cv2.waitKey(10) & 0xFF == ord('n'):
                                detection_real = False
                                break
                                
                          
                            ii += 1
                        
                        cv2.destroyAllWindows()
                        np.save(Path(trial_save,f'{trial_string}_good_detection_cell_{ce}.npy'),detection_real)
                        
                    detected_frame.loc[detections,'correct'] = str(detection_real)