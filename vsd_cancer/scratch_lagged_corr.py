#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 14:32:24 2021

@author: peter
"""
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

import itertools

from vsd_cancer.functions import correlation_functions as corrf

import scipy.interpolate as interp

top_dir = Path('/home/peter/data/Firefly/cancer')

data_dir = Path(top_dir,'analysis','full')



all_trains = np.load(Path(data_dir,'all_spike_trains.npy'),allow_pickle = True)

#Plot an example - shows effect of binsize
binsize = 10

_,corrs = corrf.get_speed_distribution(all_trains,binsize,shuffle = False)

corrs,dists,times = corrs
#interpolate onto the same xy
inv_speeds = np.abs([times[x]/dists[x] for x in range(len(times))])


interp_out = np.linspace(0.1,5,100)

interped = np.zeros((len(dists),len(interp_out)-1))

for idx in range(len(dists)):
    
    interp = np.histogram(inv_speeds[idx],weights = corrs[idx],bins = interp_out)[0]
    
    interped[idx,:] = interp
    

plt.plot(interp_out[1:],np.sum(interped,0))



'''

nulls = np.empty((0,3))
for i in range(50):
    data_null,_ = corrf.get_speed_distribution(all_trains,binsize,shuffle = True)
    nulls = np.append(nulls,data_null,axis = 0)
    

max_speed = 60
data = data[data[:,0] != 0,:]
data = data[1/np.abs(data[:,0]) < max_speed,:]

data_null = data_null[data_null[:,0] != 0,:]
data_null = data_null[1/np.abs(data_null[:,0]) < max_speed,:]



bins = np.histogram(1/np.abs(np.concatenate([data[:,0],data_null[:,0]])),bins = 50)[1]


hist = np.histogram(np.abs(1/data[:,0]),bins = bins,density = True)
hist_null = np.histogram(np.abs(1/data_null[:,0]),bins = bins,density = True)


plt.plot(bins[1:],np.log(hist[0]))
plt.plot(bins[1:],np.log(hist_null[0]))


plt.show()
plt.plot(hist[0]/hist_null[0])

'''