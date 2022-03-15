#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:29:47 2021

@author: peter
"""
import numpy as np
import pandas as pd

from pathlib import Path

import shutil


HPC = False
top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None

initial_df = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

save_dir = Path(top_dir,'analysis','full','ratio_stacks')

to_dir = Path(top_dir,'analysis','full','ratio_stacks','old_user_input')


raise ValueError('WARNING"!! THIS WILL DELETE USER INPUT ABOUT GOOD DETECTIONS!!')


user_inputs = [x for x in Path(save_dir).glob('./**/*good_detection_cell*') if 'old_user_input' not in str(x)]

#it actually moves it - but will overwrite old moves
for x in user_inputs:
    shutil.move(x,Path(to_dir,x.stem+'.npy'))