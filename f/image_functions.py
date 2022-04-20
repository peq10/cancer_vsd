#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:05:52 2020

@author: peter
"""
from . import general_functions as gf
from . import plotting_functions as pf

import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy.ndimage as ndimage
import skimage.draw
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

def to_df(stack,offset = 0):
    slopes,intercept,_ = gf.stack_linregress(stack)
    bck = slopes*np.arange(stack.shape[0])[:,None,None] + intercept
    return 100*(stack-bck)/(bck - offset),slopes,intercept


def get_roi(im,overlay = None):
    fig,ax = plt.subplots()
    fig.suptitle('Click polygon roi vertices. Middle mouse to finish.')
    ax.imshow(im)
    if overlay is not None:
        ax.imshow(overlay,cmap = pf.cust_colormap())
    fig.canvas.manager.window.showMaximized()
    fig.canvas.manager.window.raise_()
    pts = np.asarray(plt.ginput(-1, timeout=-1))
    plt.close(fig.number)
    roi_mask = skimage.draw.polygon2mask(im.shape,pts[:,::-1])
    return roi_mask.astype(int), pts

def get_cell_rois(im,num_rois):
    roi_pts = []
    roi_masks = []
    overlay = np.zeros_like(im)
    for i in range(num_rois):
        mask,pts = get_roi(im,overlay = np.ma.masked_less(overlay,1))
        overlay += np.logical_xor(mask,ndimage.morphology.binary_dilation(mask)).astype(int)
        roi_pts.append(pts)
        roi_masks.append(mask)
    return roi_masks, roi_pts


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
    

def get_keyboard_input(arr):
    raise NotImplementedError('because of global  needs to be in script directly called - fix!')
    app = pg.mkQApp()
    win = KeyPressWindow()
    win.sigKeyPress.connect(keyPressed)
    pl = win.setImage(arr)
    app.exec_()
