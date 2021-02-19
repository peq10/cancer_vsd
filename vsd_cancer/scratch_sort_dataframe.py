#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 08:46:35 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

top_dir = Path('/home/peter/data/Firefly/cancer')

initial_df = Path(top_dir,'analysis',f'long_acqs_20210216_experiments_correct.csv')

df = pd.read_csv(initial_df)


for data in df.itertuples():
    if '2021' not in data.tif_file:
        continue
    else:
        break
    
    s = Path(data.tif_file)
    trial_string = '_'.join(Path(s).parts[Path(s).parts.index('cancer'):-1])
    
    date = 