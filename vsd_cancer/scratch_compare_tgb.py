#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 14:15:05 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats
import scipy.ndimage as ndimage

import astropy.visualization as av

from pathlib import Path

import f.plotting_functions as pf

import tifffile

import cv2

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

#figsave = Path(Path.home(),'Dropbox/papers/cancer/v1/TTX_washout/')
#if not figsave.is_dir():
#    figsave.mkdir()

df = df[(df.use == 'y') & ((df.expt == 'MCF10A')|(df.expt == 'MCF10A_TGFB'))]
#df = df[(df.use == 'y') & ((df.expt == 'standard')|(df.expt == 'TTX_10um'))]

trial_string = df.iloc[0].trial_string
n_thresh = len(np.load(Path(Path(save_dir,'ratio_stacks',trial_string),f'{trial_string}_event_properties.npy'),allow_pickle = True).item()['events'])

currents  = [[[],[]] for x in range(n_thresh)]
lengths  = [[[],[]] for x in range(n_thresh)]

detected_frame = pd.DataFrame()
detections = 0
use_idx = 1
#here we do not differentiate buy slip, just using all cells
for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    #print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
    seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
    cell_ids = np.arange(results['events'][0]['tc_filt'].shape[0])
    cell_ids = [x for x in cell_ids if x not in results['excluded_circle']]

    for idx,thresh_level_dict in enumerate(results['events']):
    
        
        
        event_props = results['events'][idx]['event_props']
        
        
        if data.expt == 'MCF10A':
            idx2 = 0
        elif data.expt == 'MCF10A_TGFB':
            idx2 = 1
        else:
            raise ValueError('oops')
        
        observations = [results['observation_length'][idx][x] for x in cell_ids]
        sum_current = [np.sum(np.abs(event_props[x][:,-1])) if x in event_props.keys() else 0 for x in cell_ids]
        
        #manually check finds
        if idx == use_idx:
            if np.any(np.array(sum_current)!=0):
                vid = tifffile.imread(Path(f'/media/peter/bigdata/Firefly/cancer/analysis/full/tif_viewing/grey_videos/new',f'{trial_string}_overlay_2.tif'))
                
                active_cells = [x for x in results['events'][idx] if type(x)!= str]
                locs = np.round([ndimage.center_of_mass(seg == x+1) for x in active_cells]).astype(int)
                times = [results['events'][idx][x] for x in active_cells]
                for idxxx,ce in enumerate(active_cells):
                    detected_frame.loc[detections,'trial_string'] = trial_string
                    detected_frame.loc[detections,'cell_id'] = ce
                    detected_frame.loc[detections,'loc'] = str(locs[idxxx])
                    detected_frame.loc[detections,'starts'] = str(times[idxxx][0,:]/2)
                    detections+=1
                    #also make a small video around cell

                    event_vid = vid[max(times[idxxx][0,0]//2-20,0):times[idxxx][1,-1]//2+20,max(locs[idxxx][0]-100,0):locs[idxxx][0]+100,max(locs[idxxx][1]-100,0):locs[idxxx][1]+100]
                    ii = 0
                    while True:
                        
                        # Display the resulting frame
                        cv2.imshow('Frame', event_vid[ii%event_vid.shape[0]])
                       
                        # Press Q on keyboard to  exit
                        if cv2.waitKey(25) & 0xFF == ord('y'):  
                            detection_real = True
                            break
                        elif cv2.waitKey(25) & 0xFF == ord('n'):
                            detection_real = False
                            break
                            
                      
                        ii += 1
                    detected_frame.loc[detections,'correct'] = str(detection_real)
                                        
                    
        
        currents[idx][idx2].extend(sum_current)
        lengths[idx][idx2].extend(observations)


detected_frame.to_csv(Path(top_dir,'analysis','full/MCF10A','20210522_checking10A_detections.csv'))


mcf_current = [np.array(p[0]) for p in currents if len(p[0]) != 0]
mcf_length = [np.array(x[0]) for x in lengths]
tgf_current = [np.array(p[1]) for p in currents if len(p[1]) != 0]
tgf_length = [np.array(x[1]) for x in lengths]


#adjust for observation length
mcf_adj = [mcf_current[i]/mcf_length[i] for i in range(len(mcf_current))]
tgf_adj = [tgf_current[i]/tgf_length[i] for i in range(len(tgf_current))]

#threshold level
idx = 2

#normalise the integrals to 1 max
ma = np.max([mcf_adj[idx].max(),tgf_adj[idx].max()])
mcf_adj[idx] /= ma
tgf_adj[idx] /= ma


#first plot the median, IQR of non-zero and proportion of zero current cells

nonzer_curr = [mcf_adj[idx][mcf_adj[idx]!=0],tgf_adj[idx][tgf_adj[idx]!=0]]
all_curr = [mcf_adj[idx],tgf_adj[idx]]

medians = np.array([np.median(x) for x in nonzer_curr])
IQRs = np.array([[np.percentile(x,25),np.percentile(x,75)] for x in nonzer_curr])

num_zers = np.array([np.sum(mcf_adj[idx]==0),np.sum(tgf_adj[idx]==0)])
num_cells_tot = np.array([len(mcf_adj[idx]),len(tgf_adj[idx])])

proportion_zero = num_zers/num_cells_tot


fig1,ax1 = plt.subplots()
ax1.plot(np.arange(2),medians)
ax1.fill_between(np.arange(2),IQRs[:,0],IQRs[:,1],alpha = 0.5)
ax1b = ax1.twinx()
ax1b.plot(np.arange(2),proportion_zero)



n_bins = 'knuth'
plt.cla()

density = True
cumulative =  True
log = True

fig,ax1 = plt.subplots()

linewidth = 3
mcfs_hist = av.hist(mcf_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative,color = 'r',log = log,linewidth = linewidth,label = 'MCF10A')
#mcfs_hist = av.hist(mcf_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False,color = 'r')
tgf_hist = av.hist(tgf_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative, color = 'b',log = log,linewidth = linewidth,label = 'TGFB')
#tgf_hist = av.hist(tgf_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False, color = 'b')
#wash_hist = av.hist(wash_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False,color = 'g')
plt.legend(loc = (0.5,0.1),frameon = False,fontsize = 14)
ax1.set_xlabel('Integrated activity activity per cell (a.u.)')
ax1.set_ylabel('Cell fraction')
pf.set_all_fontsize(ax1, 14)
pf.set_thickaxes(ax1, 3)
ax1.tick_params(which = 'minor',width = 3,length = 3)
ax1.set_yticks([0.9,1])
ax1.set_ylim([0.9,1])
ax1.minorticks_off()



'''
b = []

mcf_tc = []
tgf_tc = []
for data in df.itertuples():
    trial_string = data.trial_string
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    #bright =  np.load(Path(trial_save,f'{trial_string}_mean_brightness.npy'))
    #b.append(bright)
    tc = np.load(Path(trial_save,f'{trial_string}_all_tcs.npy'))
    if tc.shape[-1]!= 4999:
        continue
    mean_tc = np.mean(tc,0)
    if data.expt == 'MCF1':
        mcf_tc.append(tc/mean_tc)
    else:
        tgf_tc.append(tc/mean_tc)
        
mcf_tc = np.concatenate(mcf_tc,axis = 0)
tgf_tc = np.concatenate(tgf_tc,axis = 0)

sig = 3
mcf_tc = ndimage.gaussian_filter1d(mcf_tc,sig)
tgf_tc = ndimage.gaussian_filter1d(tgf_tc,sig)



ma = 0.1
#remove anything +/- more than 10%
wh = np.where(np.logical_or(mcf_tc > 1+ma,mcf_tc <1-ma))
mcf_tc[wh[0],wh[1]] = 1


wh2 = np.where(np.logical_or(tgf_tc > 1+ma,tgf_tc <1-ma))
tgf_tc[wh2[0],wh2[1]] = 1

'''