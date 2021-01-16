#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:10:38 2021

@author: peter
"""
#look at excluding ROIs outside the central column

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import cancer_functions as canf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)


center = 246,256
radius = 200

rois = []
for idx,data in enumerate(df.itertuples()):
    fname = Path(data.tif_file)
    meta = canf.load_tif_metadata(fname)
    roi = meta['FrameKey-0-0-0']['ROI']
    rois.append(np.array(roi.split('-')).astype(int))
    

rois = np.array(rois)

#now specify x,y position
circle_roi_centers = np.array(center) - rois[:,:2]


roi_df = df.loc[:,['trial_string']]
roi_df['capture_roi_x'] = rois[:,0]
roi_df['capture_roi_y'] = rois[:,1]
roi_df['capture_roi_width'] = rois[:,2]
roi_df['capture_roi_height'] = rois[:,3]
roi_df['circle_roi_center_x'] = circle_roi_centers[:,0]
roi_df['circle_roi_center_y'] = circle_roi_centers[:,1]
roi_df['circle_roi_radius'] = radius
roi_df.to_csv(Path(save_dir,'roi_df.csv'))


#now interpret the roi and make circle roi


#translate into a circle for each one

