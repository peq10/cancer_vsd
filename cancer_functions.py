#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:12:06 2020

@author: peter
"""
import numpy as np
from pathlib import Path
import shutil
import json
import re
import tifffile
import quantities as pq
import scipy.interpolate as interp


import f.general_functions as gf
import f.ephys_functions as ef

def load_tif_metadata(fname):
    fname = Path(fname)
    metadata_file = Path(fname.parent,Path(fname.stem).stem+'_metadata.txt')
    to_file = Path('/tmp/tmp_metadata.txt')
    shutil.copy(metadata_file,to_file) #this is to deal with a wierd bug due to NTFS filesystem?
    with open(to_file) as f:
        metadict = json.load(f)
    return metadict

def parse_time(metadata_time):
    date = re.sub('-','',metadata_time[:10])
    time = re.sub(':','',metadata_time[11:19])
    return date,time
    
def lin_time(time):
    return int(time[:2])*60**2 + int(time[2:4])*60 + int(time[4:])
    

def get_stack_offset(fname,ephys_start):
    date,time = parse_time(load_tif_metadata(fname)['Summary']['StartTime'])
    
    if int(date) != int(ephys_start[0]):
        raise ValueError('Date mismatch!')

    offset = lin_time(time) - lin_time(str(ephys_start[1]))
    if offset < 0:
        raise ValueError('Time mismatch!')
    
    return offset

def load_ephys(fname):
    ephys_start = ef.get_ephys_datetime(fname)
    ephys = ef.load_ephys(fname)
    cam = ephys.segments[0].events[1].times.magnitude
    Vm_vc = ephys.segments[0].analogsignals[2][:,1]
    Im_vc = ephys.segments[0].analogsignals[1]
    Vm_cc = ephys.segments[0].analogsignals[2][:,0]
    Im_cc = ephys.segments[0].analogsignals[3]
    
    return {'ephys_start':ephys_start,'ephys':ephys,'cam':cam,'Vm_vc':Vm_vc,'Im_vc':Im_vc,'Vm_cc':Vm_cc,'Im_cc':Im_cc}

def slice_cam(cam_frames,n_frames,n_repeats,T):
    starts = np.where(np.concatenate(([1], np.diff(cam_frames) > 2*T)))[0]
    #remove any consecutive and take last
    
    starts = starts[np.concatenate((~(np.diff(starts)==1),[True]))]
    sliced_frames = np.zeros((n_repeats,n_frames))
    for idx in range(n_repeats):
        st = starts[idx]
        sliced_frames[idx,...] = cam_frames[st:st+n_frames]   
        
    if np.any(np.diff(sliced_frames,axis = -1) > 2*T):
        raise ValueError('Frames not sliced properly')
    return sliced_frames

def slice_ephys(analog_signal, single_cam):
    idx0 = ef.time_to_idx(analog_signal, single_cam[0]*pq.s)
    idx1 = ef.time_to_idx(analog_signal, single_cam[-1]*pq.s)
    return analog_signal[idx0:idx1]

def slice_all_ephys(analog_signal,sliced_cam):
    all_ephys = []
    sh = len(analog_signal)
    for ca in sliced_cam:
        ep = slice_ephys(analog_signal,ca)
        if len(ep) < sh:
            sh = len(ep)
        all_ephys.append(ep)
    return np.array([np.squeeze(all_ephys[i][:sh]) for i in range(len(all_ephys))])

def get_steps_image_ephys(im_dir,ephys_fname):
    ephys_dict = load_ephys(ephys_fname)
        
    files = [f for f in Path(im_dir).glob('./**/*.tif')]
    offsets = np.array([get_stack_offset(f,ephys_dict['ephys_start']) for f in files])
    
    offsets,files = gf.sort_zipped_lists([offsets,files])
    
    for idx,f in enumerate(files):
        stack = tifffile.imread(f)
        
        if idx == 0:
            stacks = np.zeros(((len(files),)+stack.shape),dtype = np.uint16)
            
        stacks[idx,...] = stack
    
    metadata = load_tif_metadata(files[0])
    T = float(metadata['FrameKey-0-0-0']['HamamatsuHam_DCAM-Exposure'])*10**-3
    
    cam = ephys_dict['cam']
    cam = cam[np.logical_and(cam > offsets[0]-10,cam < offsets[-1] + stacks.shape[1]*T + 10)]
    
    sliced_cam = slice_cam(cam,stacks.shape[1],stacks.shape[0], T)
    
    ephys_dict['sliced_cam'] = sliced_cam
    ephys_dict['cam'] = cam
    
    if np.any(np.diff(sliced_cam[:,0] - offsets) > stacks.shape[1]*T):
        raise ValueError('Problemo!')
        
    #now slice the ephys from the cam
    for key in ['Vm_vc','Vm_cc','Im_vc','Im_cc']:
        ephys_dict[key + '_sliced'] = slice_all_ephys(ephys_dict[key],sliced_cam)
        idx0 = ef.time_to_idx(ephys_dict[key], offsets[0] - 10)
        idx1 = ef.time_to_idx(ephys_dict[key], offsets[-1] + 10)
        ephys_dict[key] = ephys_dict[key][idx0:idx1]
        
    
    return ephys_dict,stacks


def interpolate_stack(ratio_stack, framelim = 1000):
    nits = int(np.ceil(ratio_stack.shape[0]/framelim))
    
    full_res = np.zeros((2,)+ratio_stack.shape)
    for it in range(nits):
        stack = ratio_stack[it*framelim:(it+1)*framelim,...]
        result = np.zeros((2,)+stack.shape)
        y, x = np.arange(stack.shape[1],dtype = int),np.arange(stack.shape[2],dtype = int)
        z = [np.arange(0,stack.shape[0],2,dtype = int),np.arange(1,stack.shape[0],2,dtype = int)]
        for i in range(2):
            j = np.mod(i+1,2)
            result[i,i::2,...] = stack[i::2,...]
            interped = interp.RegularGridInterpolator((z[i],y,x),stack[i::2,...],bounds_error= False,fill_value=None)
            pts = np.indices(stack.shape,dtype = int)[:,j::2,...].reshape((3,-1))
            result[i,j::2,...] = interped(pts.T).reshape(stack[1::2,...].shape)
        
        full_res[:,it*framelim:it*framelim+result.shape[1],...] = result
        
    return full_res


        