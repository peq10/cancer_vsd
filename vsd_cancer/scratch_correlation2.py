#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 10:01:02 2021

@author: peter
"""
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from vsd_cancer.functions import correlation_functions as corrf
    

top_dir = Path('/home/peter/data/Firefly/cancer')

data_dir = Path(top_dir,'analysis','full')

all_trains = np.load(Path(data_dir,'correlation','all_spike_trains.npy'),allow_pickle = True)

#Plot an example - shows effect of binsize
binsize = 10

idx1 = np.argmax([len(x) for x in all_trains])
print('Should I be taking the absolute value??')
if True:
    for idx in [idx1]:
        spike_trains = all_trains[idx]
        if len(spike_trains) == 1:
            continue
        
        trains,positions,cell_ids = corrf.get_trains(spike_trains)
        binned_spikes = corrf.bin_times(trains,binsize)
        
        plt.plot(np.sum(binned_spikes,0))
        plt.show()
        
        plt.imshow(binned_spikes)
        plt.axis('auto')
        corrf.plot_raster(trains,binsize = binsize)
        
        if False:
            plt.show()
            [np.random.shuffle(x) for x in binned_spikes]
            plt.imshow(binned_spikes)
            plt.axis('auto')
        plt.show()
    

if False:      
    for binsize in np.arange(1,60,10):
        all_pairwise = corrf.get_all_pairwise(all_trains,binsize)
        plt.hist(all_pairwise,bins = 50,log = True)
        plt.show()
   

if False:
    all_pairwise = corrf.get_all_pairwise(all_trains,binsize)
    all_null = corrf.get_null_hypoth(all_trains,binsize,repeats = 10)
    plt.hist(all_pairwise,bins = 50)
    plt.show()
    plt.hist(all_null,bins = 50)
    plt.show()
    
    print(f'Mean us: {np.mean(all_pairwise)}, mean null: {np.mean(all_null)}')
    
if True:
    binsizes = np.logspace(np.log10(1),np.log10(100),20, base = 10)

    CIs= []
    resamplings = []
    resamplings_null = []

    for binsize in binsizes:
        CI,resamp,resamp_null = corrf.get_ratio_corr_CIs(all_trains,binsize,5,bootnum = 10**3)
        
        CIs.append(CI)
        resamplings.append(resamp)
        resamplings_null.append(resamp_null)
        
        

    
    np.save(Path(data_dir,'correlation','CIs_ratio.npy'),CIs)
    np.save(Path(data_dir,'correlation','resamplings_ratio.npy'),resamplings)
    np.save(Path(data_dir,'correlation','resamplings_null_ratio.npy'),resamplings_null)
