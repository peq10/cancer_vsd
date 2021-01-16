#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:10:34 2020

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import f.ephys_functions as ef
import f.general_functions as gf
import f.image_functions as imf
import f.plotting_functions as pf
import scipy.ndimage as ndimage
import tifffile
from pathlib import Path
import json
import shutil
import datetime
import re

ephys_file = '/home/peter/data/Firefly/cancer/20201022/slip2/cell1/HIGH_K_EPHYS.smr'

im_dir = '/home/peter/data/Firefly/cancer/20201022/slip2/cell1/high_K/'

#ephys = ef.load_ephys(ephys_file)
ephys_start = ef.get_ephys_datetime(ephys_file)#
ephys = ef.load_ephys(ephys_file)
cam = ephys.segments[0].events[1].times.magnitude
Vm_vc = ephys.segments[0].analogsignals[2][:,1]
Im_vc = ephys.segments[0].analogsignals[1]
Vm_cc = ephys.segments[0].analogsignals[2][:,0]
Im_cc = ephys.segments[0].analogsignals[3]

T = 0.01

im_files = Path(im_dir).glob('./**/*.tif')




def load_df_stack(fname):
    stack = tifffile.imread(str(fname))
    df_stack,slopes,intercept = imf.to_df(stack[:-1,...],16*90)
    im = np.mean(stack,0)
    return stack,df_stack,im,slopes,intercept

def get_roi(im,thresh):
    outline_roi = im > thresh
    lab_im,labels = ndimage.measurements.label(outline_roi)
    sz = np.array([np.sum(lab_im == i) for i in range(labels+1)])
    zer_size = np.sum(outline_roi == 0)
    sz[sz == zer_size] = 0
    lab = np.argmax(sz)
    x,y = ndimage.find_objects(lab_im == lab)[0]
    return x,y
    

def slice_cam(cam_frames,n_frames,T):
    starts = np.where(np.concatenate(([1], np.diff(cam_frames) > 2*T)))[0]
    #remove any consecutive and take last
    starts = starts[np.concatenate((~(np.diff(starts)==1),[True]))]
    sliced_frames = np.zeros((len(starts),n_frames))
    for idx,st in enumerate(starts):
        sliced_frames[idx,...] = cam_frames[st:st+n_frames]   
    return sliced_frames

def get_frame_times(offset,sliced_frames,delay = 7):
    idx = np.argmin(np.abs(sliced_frames[:,0] - offset - delay))
    return sliced_frames[idx,:]


offsets = []
files = []

for idx,f in enumerate(im_files):
    offsets.append(get_stack_offset(f,ephys_start))
    files.append(f)

offsets,files = gf.sort_zipped_lists([offsets,files])
offsets = np.array(offsets)
sliced_frames = slice_cam(cam,200,T)

stack = tifffile.imread(files[0])
im = np.mean(stack,0)
x_roi,y_roi = get_roi(im,5000)
im_tst = im[x_roi,y_roi]

plt.cla()
plt.imshow(im[x_roi,y_roi])

cell_roi = (np.logical_and(im_tst > 5000,im_tst < 40000)).astype(int)
outline = np.logical_xor(cell_roi,ndimage.morphology.binary_dilation(cell_roi)).astype(int)
plt.imshow(np.ma.masked_less(outline,1),cmap = pf.cust_colormap())
cell_mask = np.ma.masked_less(cell_roi,1)

stacks = np.zeros((len(files),stack.shape[0])+im_tst.shape,dtype = np.uint16)
df_stacks = np.zeros_like(stacks).astype(float)
for idx,f in enumerate(files):
    stacks[idx,...] = tifffile.imread(f)[:,x_roi,y_roi]
    df_stacks[idx,...] = imf.to_df(stacks[idx,...],16*90)[0]

mean_bright = np.mean(np.mean(df_stacks*cell_mask,-1),-1)
exclude = (mean_bright.max(-1) - mean_bright.min(-1) > 2.5).data
frame_times = np.array([get_frame_times(t,sliced_frames) for t in offsets])

mean_bright = np.mean(np.mean(stacks*cell_mask,-1),-1)


#correct for bleaching
mean_bleach = np.mean(mean_bright[~exclude,-1]-mean_bright[~exclude,0])/200
mean_bright = mean_bright.ravel() + np.arange(len(mean_bright.ravel()))*(-1*mean_bleach)
mean_bright = np.reshape(mean_bright,(52,200))

mean_bright = mean_bright[~exclude,:].ravel()
frame_times = frame_times[~exclude,:].ravel()
#masked_stacks = df_stacks*cell_mask

#mean_bright = gf.trailing_dim_mean(masked_stacks)

plt.cla()
plt.plot(Vm_cc.times[::40],Vm_cc.magnitude[::40])
#plt.plot(offsets[~exclude],gf.norm(mean_bright[~exclude])*100-50,'.')
plt.plot(frame_times,gf.norm(mean_bright)*100 - 50,'.')

#stack,df_stack,im,slopes,intercept = load_df_stack(files[0])