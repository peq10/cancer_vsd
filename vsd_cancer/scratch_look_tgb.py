#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 10:46:35 2021

@author: peter
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats
import scipy.ndimage as ndimage

import astropy.visualization as av

from pathlib import Path

import f.plotting_functions as pf

import tifffile

import cv2

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')

df = pd.read_csv(Path(save_dir,'all_events_df.csv'))


df = df[df.expt == 'MCF10A']