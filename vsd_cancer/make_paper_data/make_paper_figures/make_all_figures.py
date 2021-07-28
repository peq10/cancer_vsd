#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:44:19 2021

@author: peter
"""

from pathlib import Path

top_dir = Path('/home/peter/data/Firefly/cancer')
save_dir = Path(top_dir,'analysis','full')
figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')

import figure_1
figure_1.make_figures(initial_df,save_dir,figure_dir)