#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:10:23 2020

@author: peter
"""
import numpy as np
import pyqtgraph as pg
import scipy.stats

import cancer_functions as cf

import f.image_functions as imf
import f.general_functions as gf 

im_dir = '/home/peter/data/Firefly/cancer/20201113/slip1/cell2/steps'

ephys = '/home/peter/data/Firefly/cancer/20201113/slip1/cell2/ephys.smr'



ephys_dict,stacks = cf.get_steps_image_ephys(im_dir,ephys)

stacks = stacks[...,2:,:,:]


interped = cf.process_ratio_stacks(stacks).reshape((-1,)+stacks.shape[-3:])

df_interped = np.array([gf.to_df(interped[i,...],offset = 16*90)[0] for i in range(interped.shape[0])])

interped = interped.reshape((-1,2)+stacks.shape[-3:])
df_interped = df_interped.reshape(interped.shape)


#interped -= np.mean(interped,-3)[:,:,None,:,:]


rat = interped[:,0,:,:,:]/interped[:,1,:,:,:]

df_rat = np.array([gf.to_df(rat[i,...],offset = 0)[0] for i in range(rat.shape[0])])

if True:
    roi,_ = imf.get_cell_rois(np.mean(stacks[-1,...],-3),1)
    
    
t_courses = gf.t_course_from_roi(df_interped, roi[0])
t_courses -= np.mean(t_courses[...,:25],-1)[...,None]

t_courses_rat = gf.t_course_from_roi(df_rat, roi[0])
t_courses_rat -= np.median(t_courses_rat[...,:30],-1)[...,None]


mean_F = np.mean(t_courses[:,:,45:80],-1)

vm = ephys_dict['vcVm_sliced']

mean_v = np.mean(vm[:,20000:30000],-1)


fit_blue = scipy.stats.linregress(mean_v,mean_F[:,0])
fit_green = scipy.stats.linregress(mean_v,mean_F[:,1])