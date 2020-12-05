#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:50:32 2020

@author: peter
"""

from pathlib import Path
import cancer_functions as canf


redo = False



home = Path.home()
if 'peq10' in str(home):
    HPC = True
    top_dir = Path(Path.home(),'firefly_link/cancer')
else:
    HPC = False
    top_dir = Path('/home/peter/data/Firefly/cancer')


data_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing')


if not data_dir.is_dir():
    data_dir.mkdir()


initial_df = Path(top_dir,'analysis','long_acqs_20201129_sorted.csv')


import load_all_long
processed_df, failed_df = load_all_long.load_all_long(initial_df, data_dir,redo = redo, HPC = HPC)

processed_df.to_csv(Path(data_dir,initial_df.stem+'_loaded_long.csv'))
failed_df.to_csv(Path(data_dir,initial_df.stem+'_failed_loaded_long.csv'))

if False:
    import segment_cellpose
    
    if redo:
        segment_cellpose.segment_cellpose(Path(data_dir,initial_df.stem+'_loaded_long.csv'), data_dir, HPC = HPC)
    
print('Extracting time series...')
import make_all_t_courses
make_all_t_courses.make_all_tc(Path(data_dir,initial_df.stem+'_loaded_long.csv'), data_dir,redo = redo, njobs = 3, HPC = HPC)

if redo:
    import make_roi_overlays
    make_roi_overlays.make_all_overlay(Path(data_dir,initial_df.stem+'_loaded_long.csv'), data_dir, Path(viewing_dir,'rois'), HPC = HPC)

print('Detecting events...')
import detect_events
detect_events.detect_all_events(Path(data_dir,initial_df.stem+'_loaded_long.csv'),data_dir, redo = redo, njobs = 16, debug = False, HPC = HPC)