#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:52:16 2020

@author: peter
"""
from cellpose import models, io
from cellpose import plot

from pathlib import Path
import matplotlib.pyplot as plt

files =[f for f in Path('/home/peter/data/Firefly/cancer/segmentation/').glob('*.tif')]


model = models.Cellpose(gpu=False, model_type='cyto')

channels = [[0,0] for f in files]

for chan, filename in zip(channels, files):
    img = io.imread(filename)
    masks, flows, styles, diams = model.eval(img, diameter=None, channels=chan)

    # save results so you can load in gui
    io.masks_flows_to_seg(img, masks, flows, diams, filename, chan)

    # save results as png
    io.save_to_png(img, masks, flows, filename)
    
    
    # DISPLAY RESULTS

fig = plt.figure(figsize=(12,5))
plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
plt.tight_layout()
plt.show()