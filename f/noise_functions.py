# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:48:36 2018

@author: peq10
"""

import numpy as np

#try and extract a noise 
def get_noise_freq(stack):
    all_noise = np.zeros((stack.shape[0],stack.shape[1]))
    for idx, frame in enumerate(stack):
        ba = np.mean(frame,1)
        all_noise[idx,:] = ba
    
    
    all_ffts = np.zeros((all_noise.shape[0],int(all_noise.shape[1]/2+1)),dtype = np.complex64)
    
    for idx,row in enumerate(all_noise):
        f = np.fft.rfft(row - np.mean(row))
        all_ffts[idx,:] = f
    
    freqs = np.fft.rfftfreq(stack.shape[1])
    mean_fft = np.mean(np.abs(all_ffts),0)
    f_idx = np.where(mean_fft == np.max(mean_fft))[0]
    
    return freqs[f_idx]/np.max(freqs),mean_fft,all_noise,f_idx,freqs,all_ffts # return normalized frequency


def subtract_read_noise(stack,fourier_coeffs,freq,transform_length):
    '''
    a function that subtracts the oscillating noise from the data
    '''
    subtracted_stack = np.zeros_like(stack)
    for idx, frame in enumerate(stack):
        osc = (np.abs(fourier_coeffs[idx])/transform_length)*np.cos(freq*2*np.pi*np.arange(frame.shape[0])+np.angle(fourier_coeffs[idx]))
        subtracted_stack[idx,...] = frame - np.ones_like(frame)*np.expand_dims(osc,1)
    return subtracted_stack