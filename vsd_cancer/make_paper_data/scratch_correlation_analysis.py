#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:29:19 2021

@author: peter
"""
import numpy as np

import pandas as pd

from pathlib import Path

import itertools

import matplotlib.pyplot as plt

'''
Amanda - things to think about


-is the cc function binning the histogram into correct number of bins? currently bins into exact number needed, but leads to the following problems
 
    -What happens with only two time points in spike train - currently returns NaN
    -what happens when there is only one time point in both trains? I hacked to return 1
    
    -What happens when both spike trains have one spike in the same bin - when we shuffle the train in cc function it doesnt destroy the correlation!
   
but zero padding increases the correlation coeff! That seems wrong, although we have been actually observing those zeros...

I think these issues need to be discussed with Julia, and are why our shuffled histgram looks a bit funny.


Run the script for pictures



'''



def get_trains(spike_trains):
    '''
    This interprets how I saved the data - returns a list of spike trains, cell positions and cell ids
    '''
    
    trains = []
    pos = []
    ids = []
    
    for cell in spike_trains.keys():
        trains.append(spike_trains[cell][0])
        pos.append(spike_trains[cell][1])
        ids.append(cell)
        
    return trains,np.array(pos),ids



def cc(times_spikes_pre,times_spikes_post,binsize,max_time = 1000, shuffle = False):
    ''' Calculates the Pearson's correlation coefficient between two spike trains
    Inputs are:
    times_spikes_pre - array with time points in which the presynaptic neuron fired
    times_spikes_post - array with time points in which the postsynaptic neuron fired
    binsize - size of the bins for constructing spike trains from spike times (float)
    Returns:
    cc - Pearson's correlation coefficient between both spike trains (float)
    * The spike trains are created with a common time array that expand the overall minimum and maximum time stamp given.
    * Therefore, only time stamps on the window of interest should be passed to the function.
    '''
    time_bins = np.arange(min([min(times_spikes_pre),min(times_spikes_post)]),
                              max([max(times_spikes_pre),max(times_spikes_post)])+binsize,
                              binsize)
    
    spk_train_source = np.histogram(times_spikes_pre,bins=time_bins)[0]
    spk_train_target = np.histogram(times_spikes_post,bins=time_bins)[0]
    spk_train_source[spk_train_source>1] = 1
    spk_train_target[spk_train_target>1] = 1
    
    if shuffle:
        np.random.shuffle(spk_train_source)
        np.random.shuffle(spk_train_target)
    

    cc = np.corrcoef(spk_train_source,spk_train_target)[0,1]    
    if np.isnan(cc):
        if np.all(spk_train_target == spk_train_source):
            return 1
        else:
            return 0
        
    return cc

def get_all_pairwise_corrs(trains,binsize = 5,shuffle = False):
    '''
    trains is list of arrays of spike times
    
    shuffle allows you to shuffle the spikes 
    '''
    n = len(trains)
    result = np.zeros(int((n*(n-1)/2)))
    
    for idx,pair in enumerate(itertools.combinations(trains,2)):
        result[idx] = cc(pair[0],pair[1],binsize,shuffle = shuffle)
    
    return result


def calculate_all_pairwise(all_trains,binsize = 1,shuffle = False):
    
    
    all_res = []
    
    for x in all_trains:
        if len(x) == 1:
            continue
        
        trains,_,_ = get_trains(x)
        
        all_res += list(get_all_pairwise_corrs(trains,binsize = binsize, shuffle = shuffle))
        
    return all_res        
            

def plot_raster(trains,binsize = 1):
    min_time = np.min([np.min(x) for x in trains])
    for idx,t in enumerate(trains):
        plt.plot((t-min_time)/binsize,np.ones_like(t)+idx,'.')
        




def bin_times(trains,binsize):
    '''
    Bins all the spikes for the whole time 
    '''
    min_time = np.min([np.min(x) for x in trains])
    max_time = np.max([np.max(x) for x in trains])
    time_bins = np.arange(min_time,max_time+binsize,binsize)
    binned_spikes = np.array([np.histogram(x,bins = time_bins)[0] for x in trains])
    binned_spikes[binned_spikes>1] = 1
    
    return binned_spikes
    


def calculate_null_hypoth(all_trains,binsize = 5,repeats = 100):
    '''
    Re-samples the spike trains with shuffling
    repeats to increase the number 
    
    I think there is a problem with the shuffling as we have lots of singleton spikes

    '''
    res = []
    for c in range(repeats):
        res += calculate_all_pairwise(all_trains,binsize = binsize, shuffle = True)
    return res




all_trains = np.load('./all_spike_trains.npy',allow_pickle = True)


#Plot an example - shows effect of binsize
binsize = 1

idx = np.argmax([len(x) for x in all_trains])
spike_trains = all_trains[idx]

#TODO - do lagged analysis (lagged by pairwise distance, i.e. for constant speed)
trains,positions,cell_ids = get_trains(spike_trains)
binned_spikes = bin_times(trains,binsize)

plt.imshow(binned_spikes)
plt.axis('auto')
plot_raster(trains,binsize = binsize)

if False:
    plt.show()
    [np.random.shuffle(x) for x in binned_spikes]
    plt.imshow(binned_spikes)
    plt.axis('auto')


#Plot distribution of our data and our null hypothesis - I don't think this is correct
tst =  calculate_all_pairwise(all_trains,binsize = binsize, shuffle = False)
tst2 =  calculate_null_hypoth(all_trains,binsize = binsize,repeats = 3)

plt.figure()
plt.hist(np.abs(tst),bins = 50,log = True,density = True)
plt.hist(np.abs(tst2),bins = 50,log = True,density= True)


