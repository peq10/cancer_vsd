#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 10:35:44 2021

@author: peter
"""
import pandas as pd 
import numpy as np

from pathlib import Path
 


df1 = pd.read_csv(Path('/home/peter/data/Firefly/cancer/analysis/full/good_detections.csv'))
df2 = pd.read_csv(Path('/home/peter/data/Firefly/cancer/analysis/full/good_detections_from_yilin.csv'))



df1['cid'] = df1.trial_string + '_' + df1.cell_id.astype(int).astype(str)
df2['cid'] = df2.trial_string + '_' + df2.cell_id.astype(int).astype(str)



tst = [x for x in df2.cid]
tst2 = [x for x in df1.cid]

tst3 = [x for x in tst if x not in tst2]

tst4 = [x for x in tst2 if x not in tst]