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

import tifffile


def get_LED_calibration(top_dir,save_dir):
    
    fnames = ['/home/peter/data/Firefly/cancer/20201113/LED_calibration.smr','/media/peter/bigdata/Firefly/cancer/brightness_cal_20201228/calibration.smr']
    sizes = np.array([[393,430],[255,270]])
    sizes = np.pi*((sizes*10**-3)/2)**2

    dates = [20201113,20201228]
    
    for ii,fname in enumerate(fnames):
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
            
            if 'mw' in s.lower():
                mw = float(s[:s.find('mW')])
            elif 'uw' in s.lower():
                mw = float(s[:s.find('uW')])/1000
            else:
                raise ValueError()
            
            if 'green' in s.lower():
                res_green.append([mw,val])
            else:
                res_blue.append([mw,val])
            
        res_green = np.array(res_green)[:-2,:] #last one bad 
        res_blue = np.array(res_blue)
        
        if dates[ii] == 20201228:
            bl_fix = np.array([[0.1498,0.3298],[0.485,1.063],[0.603,1.326]])
            bl_fix_fit = stats.linregress(bl_fix[:,1],bl_fix[:,0])
            res_green[:,0] *= bl_fix_fit.slope


        res_green[:,0] /= sizes[ii,1]
        res_blue[:,0] /= sizes[ii,0]
        
        #res_blue[:,1] -= res_blue[:,1].min()
        #res_green[:,1] -= res_blue[:,1].min()
        
        fit_blue = stats.linregress(res_blue[:,1],res_blue[:,0])
        fit_green = stats.linregress(res_green[:,1],res_green[:,0])
        
        print(fit_blue.slope)
        print(fit_green.slope)
        
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
        
        #plt.show()
        
        np.save(Path(save_dir,f'LED_calibration_{dates[ii]}_blue.npy'),[fit_blue.slope,fit_blue.intercept])
        np.save(Path(save_dir,f'LED_calibration_{dates[ii]}_green.npy'),[fit_green.slope,fit_green.intercept])


if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    df_str = ''
    save_dir = Path(top_dir,'analysis','full')
    get_LED_calibration(top_dir,save_dir)