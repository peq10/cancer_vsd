#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 08:47:43 2020

@author: peter
"""
import pywt

import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage
import time

import ruptures as rpt

df = pd.read_csv('/home/peter/data/Firefly/cancer/analysis/long_acqs_20201129_sorted.csv')
save_dir = Path('/home/peter/data/Firefly/cancer/analysis/full')

corr = 11
for idx,data in enumerate(df.itertuples()):
    
    if idx!= corr:
        continue

    parts = Path(data.tif_file).parts
    trial_string = '_'.join(parts[parts.index('cancer'):-1])
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    print(trial_string)
    
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    
    #get rid of start and end
    tc = tc[:,500:-500]
    tc_fil = ndimage.median_filter(tc,(1,11))
    
    ids =  np.array([20, 84,91,86])
    
    #plt.plot(tc[ids-1,:].T + np.arange(len(ids))/50)
    #plt.plot(tc_fil[ids-1,:].T + np.arange(len(ids))/50)
    
signal = tc[83,:]

# change point detection
model = "rank"  # "l2", "rbf"
t0 = time.time()
algo = rpt.Pelt(model=model, min_size=3, jump=5).fit(signal)
result = algo.predict(pen=20)
print(time.time() - t0)


rpt.display(signal, result)
