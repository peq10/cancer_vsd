#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:20:35 2020

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

import f.image_functions as imf
import cancer_functions as canf
 


class KeyPressWindow(pg.ImageWindow):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene.keyPressEvent(ev)
        self.sigKeyPress.emit(ev)
        self.win.close()
        



def keyPressed(evt):
    print(evt.key())
    global key_input
    key_input = evt.key()
    

def mouseclick(evt):
    global mouse_clicks
    mouse_clicks = []
    mouse_clicks.append(evt.pos())
    

def get_keyboard_input(arr):
    app = pg.mkQApp()
    win = KeyPressWindow()
    win.sigKeyPress.connect(keyPressed)
    win.scene.sigMouseClicked.connect(mouseclick)
    pl = win.setImage(arr)
    app.exec_()

topdir = '/home/peter/data/Firefly/cancer/analysis/'

df = pd.read_csv(topdir + 'long_acqs_sorted.csv')

try: 
    activity = list(np.load(Path(topdir,'intermed_activity.npy')))
    activity = [x if x != '16777216' else '89' for x in activity]
except FileNotFoundError:
    activity = []





for idx,data in enumerate(df.itertuples()):
    
    
    parts = Path(data.tif_file).parts
    trial_string = '_'.join(parts[parts.index('cancer'):-1])
    df.loc[data.Index,'trial_string'] = trial_string
    data_dir = Path(topdir,'ratio_stacks',trial_string)
    df.loc[data.Index,'data_dir'] = str(data_dir)
    
    if len(activity) > idx:
        continue

    rat = np.load(Path(data_dir, f'{trial_string}_ratio_stack.npy'))[:,2:-2,2:-2]

    
    rat2 = ndimage.filters.gaussian_filter(rat,(3,2,2))
    

    get_keyboard_input(rat2)
    
    activity[idx] = key_input
    
    np.save(Path(topdir,'intermed_activity.npy'),activity)
    
    

activity_dict = {'89':'y','78':'n','77':'m'}

activity = [activity_dict[str(x)] if 'cancer' not in str(x) else x for x in activity]

df['Activity'] = activity

     


df.drop('Unnamed: 0',axis = 'columns', inplace=True)

df.to_csv(topdir + 'long_acqs_processed.csv')