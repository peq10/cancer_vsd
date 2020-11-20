#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:41:29 2020

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import pdb
import scipy.ndimage as ndimage
import pyqtgraph
import matplotlib.cm
import tifffile

import f.image_functions as imf
import f.plotting_functions as pf
import cancer_functions as canf
import f.general_functions as gf


topdir = '/home/peter/data/Firefly/cancer/analysis/'

df = pd.read_csv(topdir + 'long_acqs_processed.csv')

df = df[df.Activity == 'y']

mmouse = []
for idx,data in enumerate(df.itertuples()):

    data_dir = Path(topdir,'ratio_stacks',data.trial_string)

    if Path(data_dir, f'{data.trial_string}_rois.npy').is_file():
        continue
    
    try:
        rat2 = np.load(Path(data_dir, f'{data.trial_string}_ratio_stack_filtered.npy'))
    except FileNotFoundError:
        rat = np.load(Path(data_dir, f'{data.trial_string}_ratio_stack.npy'))[:,2:-2,2:-2]
        rat2 = ndimage.filters.gaussian_filter(rat,(3,2,2))
        rat2 = np.pad(rat2,((0,0),(2,2),(2,2)))
    
    np.save(Path(data_dir, f'{data.trial_string}_ratio_stack_filtered.npy'),rat2)

    im = np.load(Path(data_dir, f'{data.trial_string}_im.npy'))
    
    std_im = np.std(rat2,0)
    #app = pg.mkQApp()
    #pg.image(rat2)
    #app.exec_()

        
    try:
        rois  = np.load(Path(data_dir, f'{data.trial_string}_rois.npy'))
    except FileNotFoundError:
        rois = imf.get_cell_rois(std_im.T, 4)[0]   
        np.save(Path(data_dir, f'{data.trial_string}_rois.npy'),rois)
    
    #overlay = pf.make_colormap_rois(rois, matplotlib.cm.gist_rainbow)[0]
    
    #stack = tifffile.imread(data.tif_file)
    
    #t_courses = np.array([gf.t_course_from_roi(rat2, r) for r in rois])
