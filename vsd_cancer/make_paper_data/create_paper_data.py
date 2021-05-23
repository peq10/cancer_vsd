#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:50:32 2020

@author: peter
"""

from pathlib import Path
import pandas as pd
from vsd_cancer.functions import cancer_functions as canf
import sys
import numpy as np

redo = False

home = Path.home()
if 'peq10' in str(home):
    HPC = True
    top_dir = Path(Path.home(),'firefly_link/cancer')
    df_str = '_HPC'
    HPC_num = int(sys.argv[1]) - 1 # allows running on HPC with data parallelism
    redo = bool(sys.argv[2])
else:
    HPC = False
    top_dir = Path('/home/peter/data/Firefly/cancer')
    df_str = ''
    HPC_num = None


data_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing')


if not data_dir.is_dir():
    data_dir.mkdir()



initial_df = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')

if HPC:
    df_ = pd.read_csv(initial_df)
    print(f'Doing {df_.iloc[HPC_num].tif_file}')

print('Loading tif...')
import load_all_long
processed_df, failed_df = load_all_long.load_all_long(initial_df, data_dir,redo = False, HPC_num = HPC_num)
#the failed only works when not redoing
processed_df.to_csv(Path(data_dir,initial_df.stem+'_loaded_long.csv'))

#look at failed
failed_df = load_all_long.detect_failed(initial_df, data_dir)
failed_df.to_csv(Path(data_dir,initial_df.stem+'_failed_loaded_long.csv'))

#try to redo failed
load_all_long.load_failed(Path(data_dir,initial_df.stem+'_failed_loaded_long.csv'), data_dir)

#do no filt for wash in 
_,_ = load_all_long.load_all_long_washin(initial_df, data_dir,redo = False, HPC_num = HPC_num)


print('Segmenting...')
import segment_cellpose
segment_cellpose.segment_cellpose(initial_df, data_dir, HPC_num = HPC_num, only_hand_rois = True)

print('Making overlays...')
import make_roi_overlays
make_roi_overlays.make_all_overlay(initial_df, data_dir, Path(viewing_dir,'rois'), HPC_num = HPC_num)


print('Extracting time series...')
import make_all_t_courses
make_all_t_courses.make_all_tc(initial_df, data_dir,redo = False, njobs = 10, HPC_num = HPC_num)

import make_all_cell_free_t_courses
make_all_cell_free_t_courses.make_all_cellfree_tc(initial_df, data_dir, redo = False,HPC_num=HPC_num)

print('Extracting FOV time series...')
import make_full_fov_t_courses
make_full_fov_t_courses.make_all_FOV_tc(initial_df, data_dir, redo = False,HPC_num=HPC_num)

import get_dead_cells
get_dead_cells.make_all_raw_tc(initial_df, data_dir,redo = False, njobs = 10,HPC_num=HPC_num)


print('Getting mean brightnesses')
import get_all_brightness
get_all_brightness.get_mean_brightness(initial_df, data_dir)

#THESE NEED TO BE MADE SO AUTOMATICALLY UPDATE DATAFRAME
#import define_circle_rois
#import apply_circle_rois
#import apply_principle_use

print('Detecting events...')
import get_events
get_events.get_measure_events(initial_df,data_dir,
                              thresh_range = np.arange(2,4.5,0.5),
                              surrounds_z = 10,
                              exclude_first = 150,
                              tc_type = 'median',
                              exclude_circle = False)

print('Getting user input for good detections')
import get_all_good_detections
thresh_idx = 1
#get_all_good_detections.get_user_event_input(initial_df,data_dir,thresh_idx, redo = True)

print('Applying user input')
raise NotImplementedError('Do')

raise NotImplementedError('CHECK THE DEAD CELLS IN MCF10A DATA')

print('Finished successfully')
