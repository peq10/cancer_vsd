#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 19:55:47 2021

@author: peter
"""
#a script to get all the events

import numpy as np

import pandas as pd
from pathlib import Path

import skimage.measure


from vsd_cancer.functions import cancer_functions as canf

def get_all_shape_params(initial_df,save_dir):

    df = pd.read_csv(initial_df)
    for idx, data in enumerate(df.itertuples()):
        trial_string = data.trial_string
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        seg = np.load(Path(trial_save,f'{trial_string}_seg.npy'))

        props = skimage.measure.regionprops(seg)        
        
        di = {'cell':[],
              'area':[],
              'circularity':[],
              'roundness':[],
              'solidity':[],
              'aspect':[]}
        
        for idx,cell in enumerate(props):
            di['cell'].append(idx)
            di['area'].append(cell['area'])
            di['circularity'].append(4*np.pi*cell['area']/cell['perimeter']**2)
            di['roundness'] = 4*cell['area']/(np.pi*cell['major_axis_length'])
            di['solidity']  = cell['solidity']
            di['aspect'] = cell['eccentricity']
        
        results = pd.DataFrame(di)
        
        
        results.to_csv(Path(trial_save,f'{trial_string}_cell_morphologies.csv'))