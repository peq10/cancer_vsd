#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:30:44 2020

@author: peter
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg

import scipy.ndimage 
import scipy.signal

import f.chirp_z_transform as czt
import f.noise_functions as nf

import cancer_functions as canf

fname = '/home/peter/data/Firefly/cancer/20201113/slip1/cell3/long_acq/ratio_steps_no_emission_0.04_green_0.0157_blue_1.5_ms_int_2/ratio_steps_no_emission_0.04_green_0.0157_blue_1.5_ms_int_2_MMStack_Default.ome.tif'
stack = tifffile.imread(fname)


blue = stack[::2,...]
green = stack[1::2,...]

rat = (blue/blue.mean(0))/(green/green.mean(0)) - 1

tst = np.mean(rat,-1)

F = np.fft.rfft(tst,axis = 0)

f_rat = scipy.ndimage.gaussian_filter(rat,(5,1,1))


'''
tst = np.fft.rfft(blue - blue.mean(0),axis = 0)


def get_noise_freq(stack,pad_fac = 1):
    F = np.mean(np.mean(np.abs(np.fft.rfft(stack - stack.mean(0),n = int(stack.shape[0]*pad_fac),axis = 0)),-1),-1)
    plt.plot(F)
    idx = np.argmax(np.abs(F)[100:]) + 100
    return idx/len(F)


w0_b = get_noise_freq(blue,pad_fac = 1)
w0_g = get_noise_freq(green)

Q = 30
b,a = scipy.signal.iirnotch(w0_b,Q)
freq, h = scipy.signal.freqz(b, a, fs=2)
plt.plot(freq, 20*np.log10(abs(h)))

#tst_blue = scipy.signal.lfilter(b,a,rat,axis = 0)

#tst_w0,F = get_noise_freq(rat)
'''