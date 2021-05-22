#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:38:21 2021

@author: peter
"""
import numpy as np
import matplotlib.cm
import matplotlib

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pyqtgraph as pg

#import catch22

#import prox_tv

from pathlib import Path

import pandas as pd

from vsd_cancer.functions import cancer_functions as canf

import f.image_functions as imf

import f.plotting_functions as pf

trial_string = 'cancer_20201215_slip2_area1_long_acq_corr_long_acq_blue_0.0296_green_0.0765_heated_to_37_1'


top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/example')
if not figsave.is_dir():
    figsave.mkdir(parents = True)
    
for idx,data in enumerate(df.itertuples()):
    if data.trial_string == trial_string:
        break
    
idx = 2

trial_save = Path(save_dir,'ratio_stacks',trial_string)
    

im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
  
masks = canf.lab2masks(seg)

tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
tc -= 1
tc *= 100*100/5.1 # convert to mV using vclamp result

tc_filt = ndimage.gaussian_filter(tc,(0,3))

idx2 = 91



mask = seg == idx2 +1

com = ndimage.measurements.center_of_mass(mask)

sz = (100,100)

sl1 = slice(int(com[0] - sz[0]/2),int(com[0] + sz[0]/2))
sl2 = slice(int(com[1] - sz[1]/2),int(com[1] + sz[1]/2))

mask = mask[sl1,sl2]

out = np.logical_xor(mask,ndimage.binary_dilation(mask,iterations = 2))
out = out[...,None]*np.array([255,0,0,255])

out[-7:-4,-29:-5] = (255,255,255,255)

im = im[sl1,sl2]

fig,imax = plt.subplots()
imax.imshow(im,cmap = 'Greys_r')
imax.imshow(out)
imax.axis('off')
fig.savefig(Path(Path.home(),'Dropbox/Papers/cancer/Wellcome/im.png'),dpi = 300,bbox_inches = 'tight')

fig2,tcax = plt.subplots()
t = np.arange(tc.shape[-1])/5
tcax.plot(t,tc_filt[idx2,:],'k',linewidth = 2)
pf.plot_scalebar(tcax, 0, -30, 60, 10,thickness = 4)
tcax.axis('off')
fig2.savefig(Path(Path.home(),'Dropbox/Papers/cancer/Wellcome/tc.png'),dpi = 300,bbox_inches = 'tight')
