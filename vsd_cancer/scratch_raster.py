#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:15:56 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.spatial as spatial

import pyqtgraph as pg

from pathlib import Path

import elephant.unitary_event_analysis as ue

from vsd_cancer.functions import cancer_functions as canf
import quantities as pq
import neo.core

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

trial_string = '20201207_slip1_area1'

for data in df.itertuples():
    if trial_string in data.trial_string:
        trial_string = data.trial_string
        break
    
trial_save = Path(save_dir,'ratio_stacks',trial_string)


events = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))
raw = np.load(Path(trial_save,f'{trial_string}_raw_tc.npy'))
events = events['events'][2]

tc = events['tc_filt']



def get_raster(events,seg,bin_width = None):
    tc = events['tc_filt']
    active = [(x,[]) for x in events.keys() if type(x) != str]
    lens = []
    tts = []
    for cell in active:
        #get the time peak
        tts.append(tc[cell[0]])
        ee = events[cell[0]]
        for l in ee.T:
            cell[1].append(np.argmax(np.abs(tc[cell[0],l[0]:l[1]]-1))+l[0])
            lens.append(l[1]-l[0])
            
    spike_trains = get_neo_spike_trains(active,tc.shape[-1])
    if bin_width is None:
        bin_width = np.median(lens)
        
    n_bins = np.round(tc.shape[1]/bin_width).astype(int)
    bins = np.linspace(0,tc.shape[-1],n_bins+1)
    raster = np.zeros((len(active),n_bins),dtype = int)
    ids = []
    for idx,cell in enumerate(active):
        raster[idx] = np.histogram(cell[1],bins = bins)[0]
        ids.append(cell[0])
        
    #now get the pairwise distances
    coms = np.array([ndimage.measurements.center_of_mass(seg == x + 1) for x in ids])
    dist_mat = spatial.distance.squareform(spatial.distance.pdist(coms))
    seg2 = np.sum([(seg==x).astype(int)*i for i,x in enumerate(ids)],0)
    return ids,raster,dist_mat,tts,seg2,spike_trains
    

def get_neo_spike_trains(active,length,T = 0.2):
    t_start = 0
    t_stop = length*T
    trains = []
    for cell in active:
        trains.append(neo.core.SpikeTrain(np.array(cell[1])*0.2*pq.s,t_stop = t_stop,t_start = t_start))
    return trains
ids,raster,dist_mat,tts,seg2,spike_trains = get_raster(events,seg,bin_width = 10)


#tst = ue.jointJ_window_analysis(np.array(spike_trains)[None,...],(10*10**3)*pq.ms,(100*10**3)*pq.ms,(10*10**3)*pq.ms,np.arange(10),t_start = 0*pq.ms,t_stop = 4999*200*pq.ms)