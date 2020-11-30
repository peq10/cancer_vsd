#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:50:32 2020

@author: peter
"""

from pathlib import Path


redo = False

top_dir = Path('/home/peter/data/Firefly/cancer')
data_dir = Path(top_dir,'analysis','full')

if not data_dir.is_dir():
    data_dir.mkdir()


initial_df = Path(top_dir,'analysis','long_acqs_20201129_sorted.csv')


import load_all_long
processed_df, failed_df = load_all_long.load_all_long(initial_df, data_dir,redo = redo)

processed_df.to_csv(Path(data_dir,initial_df.stem+'_loaded_long.csv'))
failed_df.to_csv(Path(data_dir,initial_df.stem+'_failed_loaded_long.csv'))

import segment_cellpose

if redo:
    segment_cellpose.segment_cellpose(Path(data_dir,initial_df.stem+'_loaded_long.csv'), data_dir)


import make_all_t_courses
make_all_t_courses.make_all_tc(Path(data_dir,initial_df.stem+'_loaded_long.csv'), data_dir,redo = redo, njobs = 3)