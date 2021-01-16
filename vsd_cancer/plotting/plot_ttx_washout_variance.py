#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:19:49 2021

@author: peter
"""
#a script to plot the TTX 10 um trials

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats

import astropy.stats as ass
import astropy.visualization as av

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import bootstrapped.compare_functions as bs_compare

from pathlib import Path

import f.plotting_functions as pf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/TTX_washout')
if not figsave.is_dir():
    figsave.mkdir(parents = True)

df = df[(df.use == 'y') & (df.expt == 'TTX_10um_washout')]

all_tcs = [[],[],[]]

for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    if data.stage == 'pre':
        idx2 = 0
        #raise ValueError()
    elif data.stage == 'post':
        idx2 = 1
    elif data.stage == 'washout':
        idx2 = 2
    else:
        raise ValueError('oops')
        
    tcs = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    stds = np.load(Path(trial_save,f'{trial_string}_all_stds.npy'))
    
    cellfree = np.load(Path(trial_save,f'{trial_string}_cellfree_tc.npy'))
    
    if not np.isnan(data.finish_at):
        observe_to = int(data.finish_at)*5
        tcs = tcs[:,:observe_to]
        stds = stds[:,:observe_to]
        cellfree = cellfree[:observe_to]
    
    #plt.plot(cellfree)
    
    mean_sqr_act = np.sqrt(np.sum(((tcs-1)/stds)**2,-1)/tcs.shape[-1])
    #if np.any(mean_sqr_act <500):
    #    break
    
    all_tcs[idx2].append(mean_sqr_act)
    
pre = np.concatenate(all_tcs[0])
post = np.concatenate(all_tcs[1])
wash = np.concatenate(all_tcs[2])

n_bins = 'blocks'
log = True
av.hist(pre,bins = n_bins,histtype = 'step',density = True,log = log,color = 'b')
av.hist(post,bins = n_bins,histtype = 'step',density = True,log = log,color = 'g')
av.hist(wash,bins = n_bins,histtype = 'step',density = True,log = log,color = 'r')

#now bootstrap
all_trials = np.concatenate([pre,post])

bootstrap_null = ass.bootstrap(pre,bootnum = 10000,samples = None, bootfunc = np.sum) 
#bootstrap_null += ass.bootstrap(post,bootnum = 10000,samples = None, bootfunc = np.sum)
bootstrap_null += ass.bootstrap(wash,bootnum = 10000,samples = None, bootfunc = np.sum)
bootstrap_null /= len(pre) + len(wash) 
