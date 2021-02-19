#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:31:36 2021

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
initial_df = Path(top_dir,'analysis',f'long_acqs_20210216_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

trial_string = '20201215_slip2_area1_long_acq_corr'#

use = []
cfs = [] 
cfss = []
ats = []
atss = []


for data in df.itertuples():

    if not np.isnan(data.finish_at):
        observe_to = int(data.finish_at)*5
    else:
        observe_to = None

    trial_save = Path(save_dir,'ratio_stacks',data.trial_string)
    
    out_t = np.load(Path(trial_save, f'{data.trial_string}_cellfree_tc.npy'))[:observe_to]
    out_s = np.load(Path(trial_save, f'{data.trial_string}_cellfree_std.npy'))[:observe_to]
    
    all_t = np.load(Path(trial_save, f'{data.trial_string}_full_fov_tc.npy'))[:observe_to]
    all_s = np.load(Path(trial_save, f'{data.trial_string}_full_fov_std.npy'))[:observe_to]
    
    im = np.load(Path(trial_save, f'{data.trial_string}_im.npy'))
    
    if 'washin' in data.expt:
        use.append(data.use)
        cfs.append(np.std(out_t-1)*100/np.sqrt(len(out_t)))
        cfss.append(np.std(out_t))
        ats.append(np.std(all_t)/np.sqrt(len(out_t)))
        atss.append(np.std(all_s)/np.sqrt(len(out_t)))
    else:
        use.append(data.use)
        cfs.append(np.std(out_t-1)*100/np.sqrt(len(out_t)))
        cfss.append(np.std(out_t))
        ats.append(np.std(all_t)/np.sqrt(len(out_t)))
        atss.append(np.std(all_s)/np.sqrt(len(out_t)))
    
cfs = np.array(cfs)
cfss = np.array(cfss)
ats = np.array(ats)
atss = np.array(atss)

use = np.array(use)
d = {'y':0,'m':1,'n':1,'nan':0}
use2 = np.array([float(d[x]) for x in use])
use2 /= np.sum(use2)

tst = cfs/cfss
tst1 = cfs
tst2 = cfss
tst3 = ats/atss
tst4 = ats/atss
tst5 = ats/atss
tsts = [tst,tst1,tst2,tst3,tst4,tst5]
sorts = [np.argsort(x) for x in tsts]

plt.cla()
#plt.plot(cfs[sort]/cfs.max())
#plt.plot(cfss[sort]/cfss.max())
plt.plot(tsts[1][sorts[1]]/tsts[1].max())
plt.plot(use2[sorts[1]]/use2.max())
for i,s in enumerate(sorts):
    plt.plot(1- np.cumsum(use2[s]),label = i)
    
plt.legend()

thresh = np.max(np.sort(cfs)[:80])

use3 = cfs < thresh

use3 = np.logical_and(use3,~(use2>0))

plt.plot(use3[sorts[1]])

use3 = ['y' if x == 1 else 'n' for x in use3 ]

df['thresh_use'] = use3
df.to_csv(initial_df)