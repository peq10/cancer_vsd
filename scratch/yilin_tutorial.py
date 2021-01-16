#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:04:45 2021

@author: peter
"""

import numpy as np
import matplotlib.cm

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pyqtgraph as pg

from pathlib import Path

#All trials are identified by a 'trial_string' like the following.
#I have given you data from a single trial
trial_string = 'cancer_20201215_slip2_area1_long_acq_corr_long_acq_blue_0.0296_green_0.0765_heated_to_37_1'

#replace this with where you saved the data I send
data_folder = Path(Path.home(),'data/Firefly/cancer/Yilin/')

#all variables saved as folder/<<trial_string>>_save_name.npy

#the most important is this stack - is the ratio of the blue and green excitation channels and is 
#proportional to the membrane voltage
ratio_stack = np.load(Path(data_folder,trial_string,f'{trial_string}_ratio_stack.npy'))

#this is saved as a single precision float - need to convert to double before manipulating as otherwise will 
#get floating point problems COMMENT OUT IF LOW MEMORY
#ratio_stack = ratio_stack.astype(np.float64)

print(f'The voltage imaging data is saved in a 3D stack as (t,y,x). The dimensions are {ratio_stack.shape}')
print('The sampling period is (0.2 ms, 1.04 um, 1.04 um)')

#define the sampling period for later
T = 0.2

#I have not included the raw data, but you can get it if you want - just tell me.
#I include a fluorescence image of the data to show the cell segmentation.
im = np.load(Path(data_folder,trial_string,f'{trial_string}_im.npy'))

fig,ax = plt.subplots()
ax.imshow(im,cmap = 'Greys_r')
fig.suptitle('Fluorescence image from the blue channel\nat the beginning of the acquisition')


#I use cellpose (https://github.com/MouseLand/cellpose) for cell segmentation. You will need to install it
#if you want to run the segmentation. I have run it for you, so you don't have to

I_want_to_run_the_segmentation_myself = False

if I_want_to_run_the_segmentation_myself:
    from cellpose import models
    model = models.Cellpose(gpu=False, model_type='cyto')
    seg, flows, styles, diams = model.eval(im, diameter=30, channels=[0,0])
else:
    seg = np.load(Path(data_folder,trial_string,f'{trial_string}_seg.npy'))
    
    
print(f'We have segmented {seg.max()} cells')
    
def lab2masks(seg):
    '''
    This function turns a labelled segmentation into a 3D array of individual cell masks

    Parameters
    ----------
    seg : 2D int array shape (m,n)
        Segmentation where each cell labelled by an integer 1-k.
        0 where no cells are segmented

    Returns
    -------
    masks: 3D int array shape (k,m,n), ints of 0 or 1.
        Kth slice returns an array with True where the kth segmentation segmented.

    '''
    masks = []
    for i in range(1,seg.max()+1):
        masks.append((seg == i).astype(int))
    return np.array(masks)

#Generate individual ROIs
#CAREFUL!! roi[i] is where seg == i+1, as seg == 0 is no cells
masks = lab2masks(seg)


#This code is fopr generating outlines to plot
outlines = np.array([np.logical_xor(mask,ndimage.morphology.binary_dilation(mask)) for mask in masks]).astype(int)
outlines *= np.arange(outlines.shape[0]+1)[1:,None,None] 
outlines = np.ma.masked_less(outlines,1)
overlay = matplotlib.cm.gist_rainbow(np.sum(outlines,0)/outlines.shape[0])

fig,axarr = plt.subplots(ncols = 3)
fig.suptitle('Cells and their segmentation')
axarr[0].imshow(im,cmap = 'Greys_r')
axarr[1].imshow(seg,cmap = 'gist_rainbow')
axarr[2].imshow(im,cmap = 'Greys_r')
axarr[2].imshow(overlay)


#Now look at how we generate the individual cells time courses

def t_course_from_roi(nd_stack,roi):
    '''
    generates time courses from the roi given and the n-dimensional stack.
    Assumes a 2D ROI and a stack with trailing dimensions (...,t,y,x)
    '''
    if len(roi.shape) != 2:
        raise NotImplementedError('Only works for 2d ROIs')
    wh = np.where(roi)
    return np.mean(nd_stack[...,wh[0],wh[1]],-1)

#This code generates an average time course for each cell
tc = np.array([t_course_from_roi(ratio_stack,roi) for roi in masks])

#its useful to filter to remove some noise
tc_filt = ndimage.gaussian_filter(tc,(0,3))


#this code plots the individual time courses against each other 
fig,ax = plt.subplots()
ax.plot(np.arange(tc_filt.shape[1])*T,tc_filt.T + np.arange(tc_filt.shape[0])/100)
ax.set_xlabel('Time (s)')
fig.suptitle('Fluroescence time courses from all cells')


#we can see that some cells are doing interesting things - e.g. this one.
cell = 91
overlay = matplotlib.cm.gist_rainbow(np.sum(outlines[cell:cell+1,...],0)/cell)

fig = plt.figure(constrained_layout = True)
gs  = fig.add_gridspec(3,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,:])
ax4 = fig.add_subplot(gs[2,:])

ax1.imshow(im,cmap = 'Greys_r')
ax1.imshow(overlay)
ax2.imshow(masks[cell,...])
ax3.plot(np.arange(tc_filt.shape[1])*T,(tc[cell]-1)*100)
ax3.plot(np.arange(tc_filt.shape[1])*T,(tc_filt[cell] - 1)*100)


#we want to detect what this cell is doing - lets make an algorithm to do this
#we detect where the cell goes outside a threshold
def soft_threshold(arr,thresh,to = 1):
    #Thresholds towards to value
    res = np.copy(arr)
    wh = np.where(np.abs(arr - to) < thresh)
    n_wh = np.where(np.abs(arr - to) >= thresh)
    sgn = np.sign(arr - to)
    
    res[wh] = to
    res[n_wh] -= sgn[n_wh]*thresh
    
    return res


#see what happens when you change the below value!
thresh_value = 0.0025

threshed = soft_threshold(tc_filt[cell,...], thresh_value)
ax4.plot(np.arange(tc_filt.shape[1])*T,(threshed-1)*100)


#now detect the events and calculate their properties

#Detect start and end locations by finding derivative
locs = np.diff((np.abs(threshed - 1) != 0).astype(int), prepend=0, append=0)
#llocs is (num_events,2) array of event start and end indices 
llocs = np.array((np.where(locs == 1)[0], np.where(locs == -1)[0])).T

for l in llocs:
    ax4.fill_betweenx([(threshed-1).min()*100,(threshed-1).max()*100],l[0]*T,l[1]*T,facecolor = 'r',alpha = 0.5)
    ax3.fill_betweenx([(tc[cell]-1).min()*100,(tc[cell]-1).max()*100],l[0]*T,l[1]*T,facecolor = 'r',alpha = 0.5)


#now find the properties of the events
event_lengths = np.zeros(llocs.shape[0])
event_amplitudes = np.zeros_like(event_lengths)
event_integrals = np.zeros_like(event_lengths)

for idx,l in enumerate(llocs):
    event_lengths[idx] = (l[1] - l[0])*T
    event_amplitudes[idx] = tc_filt[cell,np.argmax(np.abs(tc_filt[cell,l[0]:l[1]]-1))+l[0]] - 1 
    event_integrals[idx] = np.sum(tc_filt[cell,l[0]:l[1]] - 1)


print(f'Detected {llocs.shape[0]} events in {tc.shape[1]*T:.0f} seconds, median absolute amplitude {np.median(np.abs(event_amplitudes))*100:.2f}%')