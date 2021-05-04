#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:10:10 2021

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats

import pyqtgraph as pg

from pathlib import Path

import elephant
import f.general_functions as gf
import scipy.ndimage as ndimage

from vsd_cancer.functions import cancer_functions as canf


top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'yilin_tc.csv')

df = pd.read_csv(initial_df)
df2= pd.read_csv(Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv'))


all_tcs = []
cell_id = []
lens = []

segs = []
ims = []
names = []

for data in df.itertuples():
    if data.use != 'y':
        continue
    
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',data.trial_string)
    
    ddata = df2[df2.trial_string == trial_string]
    observe_to = ddata.finish_at.values[0]
    if np.isnan(observe_to):
        observe_to = None
    else:
        observe_to = int(observe_to)
    
    results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
    tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))[:,:observe_to]
    
    segs.append(np.load(Path(trial_save,f'{trial_string}_seg.npy')))
    ims.append(np.load(Path(trial_save,f'{trial_string}_im.npy')))
    names.append(data.trial_string)
    
    events = results['events'][1]
    ids = [x for x in events.keys() if type(x) != str]
    
    for i in ids:
        if tcs.shape[-1]!= 4999:
            tcs = np.pad(tcs,((0,0),(0,4999 - tcs.shape[-1])),mode = 'constant',constant_values = np.NAN)
        all_tcs.append(tcs[i,:])
        cell_id.append(f'{trial_string}_cell_{i}')

all_tcs = np.array(all_tcs)
np.save(Path(top_dir,'analysis','yilin_tcs.npy'),all_tcs)
np.save(Path(top_dir,'analysis','yilin_tc_ids.npy'),cell_id)

np.save(Path(top_dir,'analysis','yilin_segs.npy'),segs)
np.save(Path(top_dir,'analysis','yilin_ims.npy'),ims)
np.save(Path(top_dir,'analysis','yilin_names.npy'),names)
    
    