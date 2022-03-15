#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 18:13:27 2021

@author: peter
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from vsd_cancer.functions import correlation_functions as corrf

def calculate_corrs(top_dir,save_dir, redo = False):
    
    all_trains = np.load(Path(save_dir,'correlation','all_spike_trains.npy'),allow_pickle = True)
    if redo:
        binsizes = np.logspace(np.log10(1),np.log10(100),20, base = 10)
        CIs = []
        resamplings = []
        p_vals = []
        null_dists = []
        for binsize in binsizes:
            CI,resamp = corrf.get_corr_CIs(all_trains,binsize,5,bootnum = 10**3)
            
            CIs.append(CI)
            resamplings.append(resamp)
            
            pv,nd = corrf.calculate_p_value(all_trains,binsize,bootnum = 10**3)
            p_vals.append(pv)
            null_dists.append(nd)
            print(f'Binsize: {binsize}, p = {pv}')
            
            plt.show()
            
        CIs_nd = [(np.percentile(x,5),np.percentile(x,95)) for x in null_dists]
        np.save(Path(save_dir,'correlation','CIs.npy'),CIs)
        np.save(Path(save_dir,'correlation','CIs_null.npy'),CIs_nd)
        np.save(Path(save_dir,'correlation','bootstrapped_samples.npy'),resamplings)
        np.save(Path(save_dir,'correlation','p_vals.npy'),p_vals)
        np.save(Path(save_dir,'correlation','null_dists.npy'),null_dists)
        np.save(Path(save_dir,'correlation','binsizes.npy'),binsizes)