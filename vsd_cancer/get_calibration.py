#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 18:01:54 2020

@author: peter
"""
import numpy as np

import f.ephys_functions as ef

import matplotlib.pyplot as plt
import scipy.stats as stats

from pathlib import Path

top_dir = Path('/home/peter/data/Firefly/cancer')
save_dir = Path(top_dir,'analysis','full')


def get_LED_calibration(top_dir,save_dir)
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
        
    res_green = np.array(res_green)[:-2,:] #last one bad 
    res_blue = np.array(res_blue)
    
    
    #res_blue[:,1] -= res_blue[:,1].min()
    #res_green[:,1] -= res_blue[:,1].min()
    
    fit_blue = stats.linregress(res_blue[:,1],res_blue[:,0])
    fit_green = stats.linregress(res_green[:,1],res_green[:,0])
    
    
    
    fig,ax = plt.subplots()
    ax.plot(res_blue[:,1],res_blue[:,1]*fit_blue.slope + fit_blue.intercept,'k',linewidth = 3)
    ax.plot(res_blue[:,1],res_blue[:,0],'.r', markersize = 10)
    ax.set_xlabel('Blue LED signal')
    ax.set_ylabel('Blue LED Power')
    
    
    fig,ax = plt.subplots()
    ax.plot(res_green[:,1],res_green[:,1]*fit_green.slope + fit_green.intercept,'k',linewidth = 3)
    ax.plot(res_green[:,1],res_green[:,0],'.r', markersize = 10)
    ax.set_xlabel('Green LED signal')
    ax.set_ylabel('Green LED Power')
    
    np.save(Path(save_dir,'LED_calibration_20201113_blue.npy'),[fit_blue.slope,fit_blue.intercept])
    np.save(Path(save_dir,'LED_calibration_20201113_green.npy'),[fit_green.slope,fit_green.intercept])
