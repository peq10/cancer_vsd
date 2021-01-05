#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:50:32 2020

@author: peter
"""

from pathlib import Path
import pandas as pd
import cancer_functions as canf
import sys


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



initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

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


print('Segmenting...')
import segment_cellpose
#segment_cellpose.segment_cellpose(initial_df, data_dir, HPC_num = HPC_num)

print('Making overlays...')
import make_roi_overlays
#make_roi_overlays.make_all_overlay(initial_df, data_dir, Path(viewing_dir,'rois'), HPC_num = HPC_num)


print('Extracting time series...')
import make_all_t_courses
#make_all_t_courses.make_all_tc(initial_df, data_dir,redo = False, njobs = 10, HPC_num = HPC_num)

print('Detecting events...')
import get_events
get_events.get_measure_events(initial_df,data_dir)

print('Finished successfully')
