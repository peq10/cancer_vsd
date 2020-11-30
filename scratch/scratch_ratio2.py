#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:00:27 2020

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pyqtgraph as pg
import f.general_functions as gf

import f.image_functions as imf
import f.plotting_functions as pf
import scipy.signal as signal
import scipy.ndimage as ndimage
import skimage.filters
import scipy.interpolate as interp
import cancer_functions as canf
from pathlib import Path
import pywt
import matplotlib.cm
import scipy.stats

import sklearn.decomposition

def stack_norm(stack):
    return (stack - stack.min(-1).min(-1)[:,None,None])/(stack.max(-1).max(-1)[:,None,None] - stack.min(-1).min(-1)[:,None,None])


fname = '/home/peter/data/Firefly/cancer/20201028/slip3/cell1/long_acq/ratio_steps_high_both_4x4_50_hz_1/ratio_steps_high_both_4x4_50_hz_1_MMStack_Default.ome.tif'
#fname = '/home/peter/data/Firefly/cancer/20201028/slip2/cell1/steps2_no_seal_test_bright_green/ratio_steps_high_green_green_4x4_25_hz_13/ratio_steps_high_green_green_4x4_25_hz_13_MMStack_Default.ome.tif'
#fname = '/home/peter/data/Firefly/cancer/20201028/slip3/cell2/steps/ratio_steps_high_both_4x4_100_hz_13/ratio_steps_high_both_4x4_100_hz_13_MMStack_Default.ome.tif'

fname = '/home/peter/data/Firefly/cancer/20201028/slip3/cell2/long/ratio_steps_high_both_4x4_100_hz_1/ratio_steps_high_both_4x4_100_hz_1_MMStack_Default.ome.tif'

fname = '/home/peter/data/Firefly/cancer/20201113/slip1/cell3/long_acq/ratio_steps_no_emission_0.04_green_0.0157_blue_1.5_ms_int_2/ratio_steps_no_emission_0.04_green_0.0157_blue_1.5_ms_int_2_MMStack_Default.ome.tif'
stack = tifffile.imread(fname)

    
    
if False:
    masks,pts = imf.get_cell_rois(stack[0,...],10)
else:
    masks = np.load(Path(Path(fname).parent,'masks.npy'))
    masks = masks[np.array([1,4,6,8,2])-1,...]
    
    
interped_stack = canf.interpolate_stack(stack)
  

tst = np.mean(np.mean(interped_stack*masks[None,0,...],-1),-1)

pca = sklearn.decomposition.PCA(n_components=2)
pca.fit(tst.T)
tst2 = pca.transform(tst.T)

t = np.arange(0,100,0.01)

x = np.zeros_like(t)
x[np.logical_and(t>5,t<6)]  = 1
x[np.logical_and(t>78,t<90)]  = -1

rat = x*np.ones(2)[:,None]
rat[1,:] *= -1
rat += np.random.normal(0,scale = 0.15,size = rat.shape)

rat += 8*np.random.rand(2)[:,None]

low_freq_noise = band_limited_noise(0,0.05,samples = len(x),samplerate = 100)
#add correlated noise
rat += low_freq_noise[None,:]*2000


plt.plot(t,rat.T)

vec = np.copy(rat)

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def PCA(vec):
    #channels in first dimension
    vec -= vec.mean(-1)[:,None]
    cov = np.cov(vec)
    eigval,eigvec = np.linalg.eigh(cov)
    #sort into ascending order
    eigval = eigval[::-1]
    eigvec = eigvec[:,::-1]
    if eigval[1] > eigval[0]:
        raise ValueError('I read the docs wrong clearly')
    
    pcs = np.matmul(eigvec.T,vec)
    return pcs, eigvec, eigval
    

def linear_proc(vec):
    vec -= vec.mean(-1)[:,None]
    cov = np.cov(vec)
    return 0

def fit_l1_line(vec):
    vec -= vec.mean(-1)[:,None]
    mid = 0
    ranges = np.pi/np.logspace(0,1.5,num = 5)
    for idx0,range_ in enumerate(ranges):
        angles = np.arange(-range_/2+mid,range_/2+mid,range_/25)
        res = np.zeros((2,len(angles)))
        for idx,theta in enumerate(angles):
            c,s = np.cos(theta),np.sin(theta)
            rot_mat = np.array([[c,-s],[s,c]])
            rot_vec = np.matmul(rot_mat,vec)
            res[0,idx] = np.sum(np.abs(rot_vec[0,...]))
            res[1,idx] = np.sum(np.abs(rot_vec[1,...]))
            
            #print(idx)
            #plt.plot(rot_vec.T)
            #plt.show()

        
        mid = angles[np.argmax(res[0,:]/res[1,:])]
        plt.plot(res[1,:]/res[0,:])
        #plt.show()
    

    c,s = np.cos(mid),np.sin(mid)
    rot_mat = np.array([[c,-s],[s,c]])
    rot_vec = np.matmul(rot_mat,vec)
    plt.show()
    plt.plot(rot_vec.T)
    
    return rot_vec,mid
    
rot_vec,theta = fit_l1_line(rat)
'''
blue = blue/blue.max()
green = green/green.max()

ratio = blue/green 


sz = 256,140

def trim_to_size(stack,sz):
    trim = np.divmod(stack.shape[-2:],sz)[-1]/2    
    return stack[...,int(np.ceil(trim[0])):-int(np.floor(trim[0])),int(np.ceil(trim[1])):-int(np.floor(trim[1]))]

#ratio = trim_to_size(ratio, sz)
ratio_df = (ratio - ratio.mean(0))/ratio.mean(0)

#tifffile.imsave('./ratio_tst_tif.tif',gf.to_16_bit(ratio))
#tifffile.imsave('./green_tst_tif.tif',gf.to_16_bit(green))
#tifffile.imsave('./blue_tst_tif.tif',stack[::2,...])


masks,pts = imf.get_cell_rois(green[0,...],1)

t_course = gf.t_course_from_roi(ratio_df, masks[0])
plt.plot(t_course) 
#pg.image(ratio)

#im = stack[1,...]
def process(stack):
    return signal.medfilt(gf.bin_stack(stack, 1),(1,1,1))


#tst = np.fft.fftshift(np.fft.fft(process(ratio_df),axis = 0),axes = 0)

#tst2 = pywt.wavedec(process(ratio_df),'db7',axis = 0)


#pg.image(np.abs(tst))
'''
