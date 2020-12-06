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
import scipy.ndimage as ndimage
import pandas as pd
import datetime
import pdb

import f.general_functions as gf
import f.ephys_functions as ef

def load_tif_metadata(fname):
    fname = Path(fname)
    metadata_file = Path(fname.parent,Path(fname.stem).stem+'_metadata.txt')
    if 'rds' in str(Path.home()):
        to_file = metadata_file
    else:
        to_file = Path('/tmp/tmp_metadata.txt')
        shutil.copy(metadata_file,to_file) #this is to deal with a wierd bug due to NTFS filesystem?
        
    with open(to_file,'r') as f:
        metadict = json.load(f)
    return metadict

def parse_time(metadata_time):
    date = metadata_time.split(' ')[0].split('-')
    time = metadata_time.split(' ')[1].split(':')
    return date,time
    
def lin_time(time):
    return float(time[0])*60**2 + float(time[1])*60 + float(time[2])
    

def get_stack_offset(fname,ephys_start):
    date,time = parse_time(load_tif_metadata(fname)['Summary']['StartTime'])
    
    if int(date) != int(ephys_start[0]):
        raise ValueError('Date mismatch!')

    offset = lin_time(time) - lin_time(str(ephys_start[1]))
    if offset < 0:
        raise ValueError('Time mismatch!')
    
    return offset


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
    ephys_dict = ef.load_ephys(ephys_fname)
        
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
    for key in ['vcVm','ccVm','ccIm','ccVm']:
        if key not in ephys_dict.keys():
            continue
        ephys_dict[key + '_sliced'] = slice_all_ephys(ephys_dict[key],sliced_cam)
        idx0 = ef.time_to_idx(ephys_dict[key], offsets[0] - 10)
        idx1 = ef.time_to_idx(ephys_dict[key], offsets[-1] + 10)
        ephys_dict[key] = ephys_dict[key][idx0:idx1]
        
    
    return ephys_dict,stacks


def process_ratio_stacks(stacks):
    '''
    assumes dims = (....,t,y,x)
    '''
    sh = stacks.shape
    stacks = stacks.reshape((-1,)+sh[-3:])
    res = np.zeros((stacks.shape[0],2)+sh[-3:]).astype(float)
    for idx,st in enumerate(stacks):
        res[idx,...] = interpolate_stack(st)
        
    return res.reshape(sh[:-3]+(2,)+sh[-3:])
    

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


def get_LED_powers(LED,cam,T_approx,cam_edge = 'falling'):
    #assumes LED and cam contain only sliced vals, cam is camDown
    if cam_edge != 'falling':
        raise NotImplementedError('Only implemented for cam falling edge')
    
    #do a rough pass then a second to get LED real value
    ids = ef.time_to_idx(LED, [cam[1]+T_approx,cam[1]+3*T_approx,cam[0] - T_approx, cam[0],cam[1] - T_approx, cam[1]])
    zer = LED[ids[0]:ids[1]].magnitude.mean()
    l1 = LED[ids[2]:ids[3]].magnitude.mean()
    l2 = LED[ids[4]:ids[5]].magnitude.mean()
    thr = 0.5*(zer + min(l1,l2)) + zer
    
    LED_thr = LED > thr
    
    ##get actual T
    T = (np.sum(LED_thr.astype(int))/len(cam))/LED.sampling_rate.magnitude
    
    if np.abs(T-T_approx) > T_approx/2:
        print(T)
        print(T_approx)
        print('Problems?')
    
    #now get accurate values 
    ids1 = np.array([ef.time_to_idx(LED, cam[::2] - 3*T/4),ef.time_to_idx(LED, cam[::2] - T/4)]).T
    led1 = np.mean([LED[x[0]:x[1]].magnitude.mean() for x in ids1])
    
    ids2 = np.array([ef.time_to_idx(LED, cam[1::2] - 3*T/4),ef.time_to_idx(LED, cam[1::2] - T/4)]).T
    led2 = np.mean([LED[x[0]:x[1]].magnitude.mean() for x in ids2])
    
    ids3 = np.array([ef.time_to_idx(LED, cam[1:-1:2] + T),ef.time_to_idx(LED, cam[2::2] - 5*T)]).T
    zer = np.mean([LED[x[0]:x[1]].magnitude.mean() for x in ids3])
    
    led1 -= zer
    led2 -= zer

    return led1,led2

def cam_check(cam,cam_id,times,e_start,fs):
    if cam_id+len(times) > len(cam):
        print('length issue')
        return False
    
    if len(times) % 2 ==1:
        times = times[:-1]
        
    cam_seg = cam[cam_id:cam_id+len(times)]
    IFI = np.array([np.diff(cam_seg[::2]),np.diff(cam_seg[1::2])])
    
    #check frame rate consistent
    if np.any(np.abs(IFI - 1/fs) > (1/fs)/100):
        print('IFI issue')
        return False
    
    #compare our segment with if we are off by one each direction - are we at a minimum?
    if cam_id+len(times) == len(cam):
        v = [-1,0]
    elif cam_id == 0:
        v = [0,1]
    else:
        v = [-1,0,1]
        
    var = [np.std(cam[cam_id+x:cam_id+x+len(times)]+e_start-times) for x in v]
    if var[1] != min(var):
        print('Bad times?')
        return False

    return True

def save_result_hdf(hdf_file,result_dict,group = None):
    f = hd5py.File(hdf_file,'a')
    
    if group is not None:
        group = f'{group}/{to_trial_string(result_dict["tif_file"])}'
    else:
        group = f'{to_trial_string(result_dict["tif_file"])}'
        
    grp = f.create_group(group)
    
    for key in result_dict.keys():
        t = type(result_dict[key])
        if t == 'neo.core.analogsignal.AnalogSignal':
            print(0)
        elif t == 'numpy.ndarray':
            print(1)
        else:
            raise NotImplementedError('Implement this')
    
def get_all_frame_times(metadict):
    frames = []
    times = []
    for k in metadict.keys():
        if k == 'Summary':
            continue
        
        frame = int(k.split('-')[1])
        frames.append(frame)
        time = metadict[k]['UserData']['TimeReceivedByCore']['scalar'].split(' ')[1].split(':')
        time = float(time[0])*60**2 + float(time[1])*60 + float(time[2])
        times.append(time)
        
    frames,times = gf.sort_zipped_lists([frames,times])

    return np.array(frames),np.array(times)

def load_and_slice_long_ratio(stack_fname,ephys_fname, T_approx = 3*10**-3, fs = 5):
    stack = tifffile.imread(stack_fname)

    n_frames = len(stack)
    
    if Path(ephys_fname).is_file():
        ephys_dict = ef.load_ephys_parse(ephys_fname,analog_names=['LED','vcVm'],event_names = ['CamDown'])
       
        e_start = [float(str(ephys_dict['ephys_start'][1])[i*2:(i+1)*2])  for i in range(3)]
        e_start[-1] += (float(ephys_dict['ephys_start'][2])/10)/1000
        e_start = lin_time(e_start)
        
        meta = load_tif_metadata(stack_fname)
        frames,times = get_all_frame_times(meta)
        
        cam = ephys_dict['CamDown_times']
        cam_id = np.argmin(np.abs(cam + e_start - times[0]))
    

        if not cam_check(cam,cam_id,times,e_start,fs):
            if cam_check(cam,cam_id-1,times,e_start,fs):
                print('sub 1')
                cam_id -= 1
            elif cam_check(cam,cam_id+1,times,e_start,fs):
                print('plus 1')
                cam_id += 1
            elif cam_check(cam,cam_id-2,times,e_start,fs):
                print('sub 2')
                cam_id -= 2
            else:
                
                raise ValueError('possible bad segment')

        
        cam = cam[cam_id:cam_id+n_frames]
        
        #extract LED powers (use slightly longer segment)
        idx1, idx2 = ef.time_to_idx(ephys_dict['LED'], [cam[0] - T_approx*5,cam[-1] + T_approx*5])    
        LED_power = get_LED_powers(ephys_dict['LED'][idx1:idx2],cam,T_approx)
        
        #return LED and vm on corect segment
        idx1, idx2 = ef.time_to_idx(ephys_dict['LED'], [cam[0] - T_approx, cam[-1]])
        LED = ephys_dict['LED'][idx1:idx2]
        
        idx1, idx2 = ef.time_to_idx(ephys_dict['vcVm'], [cam[0] - T_approx, cam[-1]])
        vcVm = ephys_dict['vcVm'][idx1:idx2]
        
        if LED_power[0] < LED_power[1]:
            blue = 0
        else:
            blue = 1
    
    else:
        
        blue = 0
        cam = None 
        LED = None
        LED_power = None
        vcVm = None
        ephys_fname = None
        
    ratio_stack = stack2rat(stack,blue = blue)
    
    result_dict = {'cam':cam,
                   'LED':LED,
                   'im':np.mean(stack[blue:100:2],0),
                   'LED_powers':LED_power,
                   'ratio_stack':ratio_stack, 
                   'vcVm':vcVm,
                   'tif_file':stack_fname,
                   'smr_file':ephys_fname}
    
    return result_dict


def stack2rat(stack,blue = 0,av_len = 1000,remove_first = True):
    if remove_first:
        stack = stack[2:,...]
    
    if blue == 0:
        blue = stack[::2,...].astype(float)
        green = stack[1::2,...].astype(float)
    else: #if the leds flipped
        blue = stack[1::2,...].astype(float)
        green = stack[::2,...].astype(float)
        
    #divide by mean
    blue /= ndimage.uniform_filter(blue,(av_len,0,0),mode = 'nearest')
    green /= ndimage.uniform_filter(green,(av_len,0,0), mode = 'nearest')
    
    rat = blue/green
    
    return rat


def strdate2int(strdate):
    return int(strdate[:4]),int(strdate[4:6]),int(strdate[-2:])

def select_daterange(str_date,str_mindate,str_maxdate):
    if ((datetime.date(*strdate2int(str_date)) - datetime.date(*strdate2int(str_mindate))).days >= 0) and ((datetime.date(*strdate2int(str_date)) - datetime.date(*strdate2int(str_maxdate))).days <= 0):
        return True
    else:
        return False
    
    
def get_tif_smr(topdir,savefile,min_date,max_date,prev_sorted = None,only_long = False):
    if min_date is None:
        min_date = '20000101'
    if max_date is None:
        max_date = '21000101'

    home = Path.home()
    local_home = '/home/peter'
    hpc_home = '/rds/general/user/peq10/home'
    if str(home) == hpc_home:
        HPC = True
    else:
        HPC = False
        
    files = Path(topdir).glob('./**/*.tif')
    tif_files = []
    smr_files = []
    
    for f in files:
        
        
        sf = str(f)
        loc = sf.find('cancer/')+len('cancer/')
        day = sf[loc:loc+8]
        
        #reject non-date experiment (test etc.)
        try: 
            int(day)
        except ValueError:
            continue
        
        if not select_daterange(day,min_date,max_date):
            continue
        
        if 'long' not in str(f):
            continue
        
        tif_files.append(str(f))
    
    
        #search parents for smr file from deepest to shallowest
        start = f.parts.index(day)
        for i in range(len(f.parts)-1,start+1,-1):
            direc = Path(*f.parts[:i])
            smr = [f for f in direc.glob('*.smr')]
            if len(smr) != 0:
                break
        
        smr_files.append([str(s) for s in smr])
        
    max_len = max([len(x) for x in smr_files])
    
    df = pd.DataFrame()
    
    df['tif_file'] = tif_files

    for i in range(max_len):
        files = []
        for j in range(len(smr_files)):
            try:
                files.append(smr_files[j][i])
            except IndexError:
                files.append(np.NaN)
        
        df[f'SMR_file_{i}'] = files
    
    
    #now consolidate files that were split (i.e. two tif files in same directory, one has _1 at end,
    #due to file size limits on tif file size)
    remove = []
    for data in df.itertuples():
        fname = data.tif_file 
        fname2 = fname[:fname.find('.ome.tif')]+'_1.ome.tif'
        if Path(fname2).is_file():
            df.loc[data.Index,'multi_tif'] = 1
            remove.append(df[df.tif_file == fname2].index[0])
            if Path(fname[:fname.find('.ome.tif')]+'_2.ome.tif').is_file():
                raise NotImplementedError('Not done more than one extra')
        else:
            df.loc[data.Index,'multi_tif'] = 0
        
    df = df.drop(labels = remove)
    
    
    if prev_sorted is not None:
        prev_df = pd.read_csv(prev_sorted)
        if local_home in prev_df.iloc[0].tif_file and HPC:
            mismatch = True
            root = str(Path(hpc_home,'firefly_link'))
            print('mismatch')
        elif hpc_home in prev_df.iloc[0].tif_file and not HPC:
            mismatch = True
            root = str(Path(hpc_home,'data/Firefly'))
        else:
            mismatch = False
    
        for data in prev_df.itertuples():
            if mismatch:
                tf = data.tif_file
                tf = str(Path(root,tf[tf.find('/cancer/')+1:]))
                loc = df[df.tif_file == tf].index
            else:
                loc = df[df.tif_file == data.tif_file].index
                
            for i in range(max_len):
                if i == 0:
                    if mismatch:
                        sf = data.SMR_file
                        try:
                            sf = str(Path(root,sf[sf.find('/cancer/')+1:]))
                        except AttributeError:
                            sf = np.NaN
                        df.loc[loc,f'SMR_file_{i}'] = sf
                    else:
                        df.loc[loc,f'SMR_file_{i}'] = data.SMR_file
                else:
                     df.loc[loc,f'SMR_file_{i}'] = np.NaN
    
    if only_long:
        df = df[['long_acq' in f for f in df.tif_file]]

    if np.all(np.isnan(df.SMR_file_1.values.astype(float))):

        df['SMR_file'] = df.SMR_file_0
        for i in range(max_len):
            df = df.drop(columns = f'SMR_file_{i}')
    
    df.to_csv(savefile)

    
    return df
