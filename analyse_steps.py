#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:05:47 2020

@author: peter
"""
# a script to analyse the steps data.

import numpy as np
import pandas as pd
from pathlib import Path
import tifffile
import scipy.stats

import cancer_functions as canf
import f.ephys_functions as ef
import f.general_functions as gf



# =============================================================================
# dire = '/home/peter/data/Firefly/cancer/20201228'
# day = '20201228'
# 
# def get_steps_dataframe(dire,day):
#     files = Path(dire).glob('./**/*.tif')
#     smr_files = []
#     tif_files = []
#         
#     for f in files:
#         
#         meta = canf.load_tif_metadata(f)
#         if len(meta) != 1301:
#             continue
#         
#         #search parents for smr file from deepest to shallowest
#         start = f.parts.index(day)
#         for i in range(len(f.parts)-1,start+1,-1):
#             direc = Path(*f.parts[:i])
#             smr = [f for f in direc.glob('*.smr')]
#             if len(smr) != 0:
#                 break
#         
#         smr_files.append([str(s) for s in smr])
#         tif_files.append(str(f))
#         
#         
#     max_len = max([len(x) for x in smr_files])
#         
#     df = pd.DataFrame()
#     
#     df['tif_file'] = tif_files
#     
#     for i in range(max_len):
#         files = []
#         for j in range(len(smr_files)):
#             try:
#                 files.append(smr_files[j][i])
#             except IndexError:
#                 files.append(np.NaN)
#         
#         df[f'SMR_file_{i}'] = files
#         
#     return df
# 
# df = get_steps_dataframe(dire,day)
# 
# df.to_csv('/home/peter/data/Firefly/cancer/analysis/steps_20201230.csv')
# =============================================================================


def load_steps_ephys2(stack_fname,ephys_fname):
    
    stack = tifffile.imread(stack_fname)

    n_frames = len(stack)
    
    if Path(ephys_fname).is_file():
        ephys_dict = ef.load_ephys_parse(ephys_fname,analog_names=['LED','vcVm','vcIm'],event_names = ['CamDown'])
       
        e_start = [float(str(ephys_dict['ephys_start'][1])[i*2:(i+1)*2])  for i in range(3)]
        e_start[-1] += (float(ephys_dict['ephys_start'][2])/10)/1000
        e_start = canf.lin_time(e_start)
        
        meta = canf.load_tif_metadata(stack_fname)
        frames,times = canf.get_all_frame_times(meta)
        
        cam = ephys_dict['CamDown_times']
        cam_id = np.argmin(np.abs(cam + e_start - times[0]))
    

        if not cam_check_steps(cam,cam_id,times,n_frames):
            if cam_check_steps(cam,cam_id-1,times,n_frames):
                print('sub 1')
                cam_id -= 1
            elif cam_check_steps(cam,cam_id+1,times,n_frames):
                print('plus 1')
                cam_id += 1
            elif cam_check_steps(cam,cam_id-2,times,n_frames):
                print('sub 2')
                cam_id -= 2
            else:
                
                raise ValueError('possible bad segment')

        
        cam = cam[cam_id:cam_id+n_frames]
        
        #slice all
        sliced_cam = np.reshape(cam,(13,100))
        stack = np.reshape(stack,(13,100)+stack.shape[-2:])
        
        
        T_approx = 3*10**-3
        
        #extract LED powers (use slightly longer segment)
        idx1, idx2 = ef.time_to_idx(ephys_dict['LED'], [cam[0] - T_approx*5,cam[-1] + T_approx*5])    
        LED_power = canf.get_LED_powers(ephys_dict['LED'][idx1:idx2],cam,T_approx)
        
        #return LED and vm on corect segment
        idx1, idx2 = ef.time_to_idx(ephys_dict['LED'], [cam[0] - T_approx, cam[-1]])
        LED = canf.slice_all_ephys(ephys_dict['LED'],sliced_cam)
        
        
        idx1, idx2 = ef.time_to_idx(ephys_dict['vcVm'], [cam[0] - T_approx, cam[-1]])
        vcVm = canf.slice_all_ephys(ephys_dict['vcVm'],sliced_cam)
        
        idx1, idx2 = ef.time_to_idx(ephys_dict['vcVm'], [cam[0] - T_approx, cam[-1]])
        vcIm = canf.slice_all_ephys(ephys_dict['vcIm'],sliced_cam)
        
        
        if LED_power[0] < LED_power[1]:
            blue = 0
        else:
            blue = 1
            
        
    result_dict = {'cam':cam,
           'LED':LED,
           'im':np.mean(stack[:,blue::2],0),
           'LED_powers':LED_power,
           'stack':stack, 
           'vcVm':vcVm,
           'vcIm':vcIm,
           'blue_idx':blue,
           'tif_file':stack_fname,
           'smr_file':ephys_fname}
    
    return result_dict

df = pd.read_csv('/home/peter/data/Firefly/cancer/analysis/steps_20201230_sorted.csv')

def cam_check_steps(cam,cam_id,times,n_frames):
    
    try:
        diff = cam[cam_id:cam_id+n_frames] - times
    except ValueError:
        return False
    
    if diff.max() - diff.min() < 3*10**-3:
        return True
    else:
        return False


mean_fs = []
mean_vs = []
mean_rs = []
fits = []
sens = []

for data in df.itertuples():
    
    s = data.tif_file
    trial_string = '_'.join(Path(s).parts[Path(s).parts.index('cancer'):-1])
    df.loc[data.Index,'trial_string'] = trial_string
    
    trial_save = Path('/home/peter/data/Firefly/cancer/analysis/full','steps_analysis/data',trial_string)
    
    if not trial_save.is_dir():
        trial_save.mkdir(parents = True)

    
    stack_fname = data.tif_file
    ephys_fname = data.SMR_file

    result_dict = load_steps_ephys2(stack_fname,ephys_fname)
    
    for key in result_dict.keys():
        np.save(Path(trial_save,f'{trial_string}_{key}.npy'),result_dict[key])
        
    tifffile.imsave(Path('/home/peter/data/Firefly/cancer/analysis/full','steps_analysis/ims',f'{trial_string}_im.tif'),gf.to_8_bit(result_dict['im']))
    
    _,roi  = gf.read_roi_file(Path('/home/peter/data/Firefly/cancer/analysis/full','steps_analysis/rois',f'{trial_string}_roi.roi'),im_dims = result_dict['im'].shape[-2:])
    
    stack = result_dict['stack']
    bl = result_dict['blue_idx']
    print(bl)
    #blue start is high for some reason, exclude
    stack[:,bl,...] = stack[:,bl+2,...]
    
    interped_stack = canf.process_ratio_stacks(stack)
    
    #now get the time courses
    t_courses = gf.t_course_from_roi(interped_stack, roi)
    
    #use linear fit for bleaching
    sta = np.mean(t_courses[...,:5],-1)
    sto = np.mean(t_courses[...,-5:],-1)
    m = (sto - sta)/t_courses.shape[-1]
    
    lin_fit = np.arange(t_courses.shape[-1])[None,None,:]*m[:,:,None] + sta[:,:,None]
    
    df_t = ((t_courses - lin_fit) / lin_fit)
    
    np.save(Path(trial_save,f'{trial_string}_df_tc.npy'),df_t)    
    
    stim_locs = np.array([25,49])
    
    mean_f = np.mean(df_t[...,stim_locs[0]:stim_locs[1]],-1)
    
    mean_fs.append(mean_f)
    
    dr_t = (df_t[:,0,:] + 1)/(df_t[:,1,:] +1)
    
    mean_r = np.mean(dr_t[...,stim_locs[0]:stim_locs[1]],-1)
    mean_rs.append(mean_r)
    
    
    vm = result_dict['vcVm']
    v_locs = np.round((stim_locs/t_courses.shape[-1])*vm.shape[-1]).astype(int)
    
    mean_v = np.mean(vm[:,v_locs[0]:v_locs[1]],-1)
    mean_vs.append(mean_v)
    
    
    fit_blue = scipy.stats.linregress(mean_v,mean_f[:,0])
    fit_green = scipy.stats.linregress(mean_v,mean_f[:,1])
    fit_rat = scipy.stats.linregress(mean_v,mean_r)
    
    fits.append([fit_blue,fit_green,fit_rat])

    sens.append([fit_blue.slope,fit_green.slope,fit_rat.slope])
    
mean_fs = np.array(mean_fs)
mean_vs = np.array(mean_vs)
mean_rs = np.array(mean_rs)

sens = np.array(sens)


#now plot
    

