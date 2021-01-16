#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:39:37 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tifffile
import scipy.stats as stats

import cancer_functions as canf

from pathlib import Path

import f.plotting_functions as pf


top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/high_k_washin')
if not figsave.is_dir():
    figsave.mkdir(parents = True)

df = df[(df.use == 'y') & (df.expt == 'high_k_washin')]

mean_tcs = []
washin = []
wash_start = 5000 
remove_start = 1000
remove_end = 1000
T = 0.2

for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    print(trial_string)
    
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    

    
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs_washin.npy'))
    
    washin_idx = int(data.washin_idx)
    #plt.figure()
    #plt.plot(tc.T +np.arange(tc.shape[0])/100)
    
    mean_tc = np.mean(tc,0)
    
    #align
    mean_tc = (mean_tc[remove_start - (wash_start - washin_idx):-remove_end-(wash_start - washin_idx)] - 1)*100
    
    #remove bleach
    sta,sto = np.mean(mean_tc[:100]),np.mean(mean_tc[int(wash_start/2) - 600:int(wash_start/2) - 500])
    
    slope = (sta - sto)/(int(wash_start/2) - 500)
    intercept = sta
    
    mean_tc -= np.arange(len(mean_tc))*slope + intercept
    
    mean_tcs.append(mean_tc)
    

mean_tcs = np.array(mean_tcs)


wash_loc = (int(wash_start/2) - remove_start)*T
fig,ax = plt.subplots()
ax.plot(np.arange(mean_tcs.shape[1])*T,np.array(mean_tcs).T + np.arange(np.array(mean_tcs).shape[0]),'k')
ax.plot(wash_loc,mean_tcs.shape[0] - 0.5,marker = (3,0,180),markersize = 15, mfc = 'r',mec = 'r')
ax.plot(wash_loc*np.ones(2),[-0.1,mean_tcs.shape[0] - 0.5],'r',linewidth = 2)
pf.plot_scalebar(ax, 500, 0, 100, 0.5,thickness = 4)
plt.axis('off')
fig.savefig(Path(figsave,'high_k_washin.png'),bbox_inches = 'tight',dpi = 300)

    