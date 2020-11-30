#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:04:59 2020

@author: peter
"""
import numpy as np
from pathlib import Path
import pandas as pd


import cancer_functions as canf


topdir = Path('/home/peter/data/Firefly/cancer')

savefile = Path(topdir,'analysis','failed.csv')

df = pd.read_csv(savefile)

bad = [0,4]
for idx,data in enumerate(df.itertuples()):
    if idx in bad or idx < 4:
        continue
    result_dict = canf.load_and_slice_long_ratio(data.tif_file,
                                             str(data.SMR_file),
                                             T_approx = 3*10**-3,
                                             fs = 5)
    
    