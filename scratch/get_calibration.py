#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 18:01:54 2020

@author: peter
"""
import numpy as np

import f.ephys_functions as ef


fname = '/home/peter/data/Firefly/cancer/20201113/LED_calibration.smr'
ephys = ef.load_ephys_parse(fname,analog_names=['LED'],event_names = ['Notes'])

led = ephys['LED']


times = ephys['Notes_times']

labs = ephys['Notes_labels']

res_green = []
res_blue = []

for idx,t in enumerate(times):
    t1 = ef.time_to_idx(led,t+5)
    t2 = ef.time_to_idx(led,t-5)
    
    val = np.mean(led[t2:t1].magnitude)
    
    s = labs[idx].decode()
    mw = float(s[:s.find('mW')])
    
    if 'green' in s.lower():
        res_green.append([mw,val])
    else:
        res_blue.append([mw,val])
    
res_green = np.array(res_green)
res_blue = np.array(res_blue)


res_blue[:,1] -= res_blue[:,1].min()
res_green[:,1] -= res_blue[:,1].min()