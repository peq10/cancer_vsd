#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:07:04 2020

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import scipy.ndimage as ndimage
import skimage.filters
import time
import matplotlib.cm

import cancer_functions as cf

import f.general_functions as gf 
import f.plotting_functions as pf
import f.image_functions as imf

ephys = '/home/peter/data/Firefly/cancer/20201028/slip2/cell1/ephys.smr'

im_dir = '/home/peter/data/Firefly/cancer/20201028/slip2/cell1/steps'


ephys_dict,stacks = cf.get_steps_image_ephys(im_dir,ephys)

im = np.mean(stacks[0,...],0)


if False:
    masks,pts = imf.get_cell_rois(im,7)
    
    np.save('/home/peter/data/Firefly/cancer/analysis/gap_junction/rois.npy',{'im':im,'masks':masks,'pts':pts})
else:
    saved = np.load('/home/peter/data/Firefly/cancer/analysis/gap_junction/rois.npy',allow_pickle = True).item()
    masks = saved['masks']
    pts = saved['pts']
    
    order = np.array([1,4,5,6,2,3,7])-1
    masks = np.array(masks)[order]
    
    
    
df_stacks = np.zeros_like(stacks[:,1:,...]).astype(float)
for idx,st in enumerate(stacks):
    df_stacks[idx,...] = imf.to_df(st[1:,...],offset = 90*16)[0]
    
    
#get time courses
time_courses = np.mean(np.mean(df_stacks[None,...]*np.ma.masked_less(masks,1)[:,None,None,:,:],-1),-1)


cell_idx = 0
repeat_idx = 12
#make activation map
activation = gf.stack_linregress(df_stacks[repeat_idx,...],vec = time_courses[cell_idx,repeat_idx,:])[-1]


def make_colormap_rois(masks,cmap):
    outlines = np.array([np.logical_xor(mask,ndimage.morphology.binary_dilation(mask)) for mask in masks]).astype(int)
    outlines *= np.arange(outlines.shape[0]+1)[1:,None,None] 
    outlines = np.ma.masked_less(outlines,1)
    overlay = cmap(np.sum(outlines,0)/outlines.shape[0])
    return overlay, cmap(np.arange(outlines.shape[0]+1)/outlines.shape[0])[1:]


def label_roi_centroids(ax,masks,colours,fontdict = None):
    coms = [ndimage.center_of_mass(mask) for mask in masks]
    for idx,com in enumerate(coms):
        ax.text(com[1],com[0],str(idx+1),color = colours[idx],fontdict = fontdict, ha = 'center', va = 'center')




font = 12
show_im = np.copy(im)
show_im[-20:-15,-48-10:-10] = im.max()

fig,axarr = plt.subplots(nrows = 2, ncols = 2)
axarr[0][0].set_title('Single frame')
axarr[0][0].imshow(show_im, cmap = 'Greys_r')


overlay,colours = make_colormap_rois(masks,matplotlib.cm.gist_rainbow)
axarr[0][1].set_title('Time course ROIs')
axarr[0][1].imshow(show_im, cmap = 'Greys_r')
axarr[0][1].imshow(overlay)
label_roi_centroids(axarr[0][1], masks, colours,fontdict = {'fontsize':font})

axarr[1][0].imshow(activation)
axarr[1][0].set_title('Voltage signal map')

axarr[1][1].set_title('Cell time courses')
for i in range(time_courses.shape[0]):
    axarr[1][1].plot(time_courses[i,-1,:] - 5*np.arange(time_courses.shape[0])[i],color =colours[i])
    axarr[1][1].text(-10, -5*np.arange(time_courses.shape[0])[i],str(i+1),color = colours[i],fontdict = {'fontsize':font}, ha = 'center', va = 'center')

x = 5
y = 3
pos = -5*np.arange(time_courses.shape[0])[-1]
axarr[1][1].plot([180+x,200+x],[pos-y,pos-y],'k',linewidth = 3)
axarr[1][1].plot([200+x,200+x],[pos-y,pos-y+5],'k',linewidth = 3)

pf.make_square_plot(axarr[1][1])
for ax in axarr.ravel():
    ax.axis('off')
    
fig.savefig('/home/peter/Dropbox/Papers/cancer/gap_junction.png',bbox_inches = 'tight',dpi = 300)
