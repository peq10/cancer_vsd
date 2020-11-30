#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:01:47 2020

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import f.ephys_functions as ef
import f.general_functions as gf
import f.plotting_functions as pf
import quantities as pq
from pathlib import Path
import tifffile
import datetime
import scipy.ndimage as ndimage
import scipy.stats as stats

ephys = ef.load_ephys('/home/peter/data/Firefly/cancer/20201016/im/cell2/ladder/ephys_ladder.smr')

T = 1/100

Im = ephys.segments[0].analogsignals[1]
Vm = ephys.segments[0].analogsignals[2][:,1]

cam = ephys.segments[0].events[0].times

inter = np.concatenate(([0],np.where(np.diff(cam) > 2*T)[0],[len(cam)-1]))

#get ephys
all_v = []
all_i = []
all_frames = []

for i in range(len(inter)-1):
    if i == 0:
        V_t0 = ef.time_to_idx(Vm, cam[inter[i]])
        I_t0 = ef.time_to_idx(Im, cam[inter[i]])
        all_frames.append(cam[inter[i]:inter[i+1]])
    else:
        V_t0 = ef.time_to_idx(Vm, cam[inter[i]+1])
        I_t0 = ef.time_to_idx(Im, cam[inter[i]+1])
        all_frames.append(cam[inter[i]+1:inter[i+1]])
    
    V_t1 = ef.time_to_idx(Vm, cam[inter[i+1]])
    I_t1 = ef.time_to_idx(Im, cam[inter[i+1]])

    all_v.append(Vm[V_t0:V_t1])
    all_i.append(Im[I_t0:I_t1])
       
    #plt.plot(Vm[V_t0:V_t1].magnitude)
    

#now find start frame of the stim
starts = []
stops = []

aligned = []

for idx,v in enumerate(all_v):
    vv = np.abs(v - v[0])
    
    wh = np.where(vv.magnitude > 5)[0]
    sta = v.times[wh[0]]
    sto = v.times[wh[-1]]
    
    starts.append(np.argmin(np.abs(all_frames[idx]-sta)))
    stops.append(np.argmin(np.abs(all_frames[idx]-sta)))
    
    t0 = ef.time_to_idx(v, sta - 100*pq.s*10**-3)
    t1 = ef.time_to_idx(v, sto + 100*pq.s*10**-3)
    
    aligned.append(v[t0:t1].magnitude)
    plt.plot(v[t0:t1].magnitude)
    #plt.show()


aligned_arr = []
for v in aligned:
    if len(v) < 27000:
        aligned_arr.append(np.NaN*np.ones(27994))
        continue
    aligned_arr.append(np.squeeze(v[:27994]))
    
aligned_arr = np.array(aligned_arr)


files = Path('/home/peter/data/Firefly/cancer/20201016/im/cell2/ladder/').glob('./**/*.tif')
stacks = []
steps = []
times = []
for f in files:
    f = str(f)

    stacks.append(tifffile.imread(f))
    
    with tifffile.TiffFile(f) as tif:
        meta = tif.ome_metadata
    t = meta[meta.find('Acquisition')+563-536:meta.find('Acquisition') - 536 + 571]
    t = 60*int(t[3:5]) + int(t[-2:])
    times.append(t)
    
times,stacks = gf.sort_zipped_lists([times,stacks])

stacks = np.array(stacks)


def to_df(stack,offset = 0):
    slopes,intercept,_ = gf.stack_linregress(stack)
    bck = slopes*np.arange(stack.shape[0])[:,None,None] + intercept
    return 100*(stack-bck)/(bck - offset),slopes,intercept

if False:
    df_stacks = np.zeros_like(stacks).astype(float)
    for idx,st in enumerate(stacks):
        df,_,_ = to_df(st,offset = 90*16)
        df_stacks[idx,...] = df
 
roi = (np.mean(stacks[0,...],0)> 0.25*np.max(np.mean(stacks[0,...],0) )).astype(int)
masked_roi = np.ma.masked_less(roi,0.5)
t_courses = np.mean(np.mean(df_stacks*masked_roi[None,None,...],-1),-1)

t_aligned = []

for idx,t in enumerate(t_courses):
    if t[starts[idx]-10:starts[idx]+60].shape[0] != 70:
        t_aligned.append(np.NaN*np.ones(70))
        continue
    t_aligned.append(t[starts[idx]-10:starts[idx]+60] - np.mean(t[starts[idx]-10:starts[idx]]))
    
    plt.plot(t_aligned[idx])

t_aligned = np.array(t_aligned)
t_al_bin = np.mean(t_aligned.reshape((16,35,2)),-1)

mean_v = np.mean(aligned_arr[:,5000:20000],-1) - np.mean(aligned_arr[:,:1000],-1) - 50
mean_df = np.mean(t_aligned[:,15:60],-1)

n = np.isnan(mean_v)

fit = stats.linregress(mean_v[~n],mean_df[~n])
fit_x = np.arange(-90,30,1)
fit_eval = fit_x*fit.slope +fit.intercept


im = np.mean(stacks[0,...],0)

out = np.logical_xor(ndimage.morphology.binary_dilation(roi),roi)
out_ma = np.ma.masked_less(out,0.5)

x_v =  np.arange(27994)/(40*10**3)
x_f = np.arange(35)/50

fig,axarr = plt.subplots(nrows = 2,ncols = 2)
axarr[0][0].imshow(im[200:310,200:310],cmap = 'Greys_r')
axarr[0][0].imshow(out_ma[200:310,200:310],cmap = pf.cust_colormap())
axarr[0][0].axis('off')
axarr[0][1].plot(fit_x,fit_eval,'r',linewidth = 3)
axarr[0][1].plot(mean_v,mean_df,'.k')
axarr[0][1].set_xlabel('Vm (mV)')
axarr[0][1].set_ylabel('dF/F (%)')
axarr[0][1].text(-20,0.35,f'{fit.slope*100:.1f} % per 100 mV',fontdict = {'fontsize':20})


axarr[1][0].plot(x_v,aligned_arr.T)
axarr[1][0].set_xlabel('t (s)')
axarr[1][0].set_ylabel('Vm (mV)')
axarr[1][1].plot(x_f,t_al_bin.T)
axarr[1][1].set_xlabel('t (s)')
axarr[1][1].set_ylabel('dF/F (%)')

for ax in axarr.ravel():
    pf.set_all_fontsize(ax,20)




