#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:00:27 2020

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pyqtgraph as pg
import f.general_functions as gf
import f.image_functions as imf
import f.plotting_functions as pf
import scipy.signal as signal
import scipy.ndimage as ndimage
import skimage.filters
import scipy.interpolate as interp
import cancer_functions as canf
from pathlib import Path
import pywt
import matplotlib.cm
import scipy.stats

def stack_norm(stack):
    return (stack - stack.min(-1).min(-1)[:,None,None])/(stack.max(-1).max(-1)[:,None,None] - stack.min(-1).min(-1)[:,None,None])


fname = '/home/peter/data/Firefly/cancer/20201028/slip3/cell1/long_acq/ratio_steps_high_both_4x4_50_hz_1/ratio_steps_high_both_4x4_50_hz_1_MMStack_Default.ome.tif'
#fname = '/home/peter/data/Firefly/cancer/20201028/slip2/cell1/steps2_no_seal_test_bright_green/ratio_steps_high_green_green_4x4_25_hz_13/ratio_steps_high_green_green_4x4_25_hz_13_MMStack_Default.ome.tif'
#fname = '/home/peter/data/Firefly/cancer/20201028/slip3/cell2/steps/ratio_steps_high_both_4x4_100_hz_13/ratio_steps_high_both_4x4_100_hz_13_MMStack_Default.ome.tif'
fname = '/home/peter/data/Firefly/cancer/20201028/slip3/cell2/long/ratio_steps_high_both_4x4_100_hz_1/ratio_steps_high_both_4x4_100_hz_1_MMStack_Default.ome.tif'
stack = tifffile.imread(fname)




interped_stack = canf.interpolate_stack(stack)

ratio = interped_stack[0,...]/interped_stack[1,...]

ratio_df = 100*(ratio - ratio.mean(0))/ratio.mean(0)
   
    
    
if False:
    masks,pts = imf.get_cell_rois(stack[0,...],10)
else:
    masks = np.load(Path(Path(fname).parent,'masks.npy'))
    masks = masks[np.array([1,4,6,8,2])-1,...]
  
masked = np.ma.masked_less(masks,1)
    
t_courses =np.mean(np.mean(ratio_df[None,...]*masked[:,None,...],-1),-1)

def downsample_t(tc,factor):
    return np.mean(tc.reshape(-1,factor),-1)

def flatten_tc(tc):

    fit = scipy.stats.linregress(np.arange(len(tc)),tc)
    return tc - np.arange(len(tc))*fit.slope  - fit.intercept
    

im = stack[1,...]

fig,ax = plt.subplots()
overlay,colours = pf.make_colormap_rois(masks,matplotlib.cm.gist_rainbow)
#ax.imshow(im,cmap='Greys_r')
ax.imshow(overlay)
pf.label_roi_centroids(ax, masks, colours,fontdict = {'fontsize':10})
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off

pf.set_thickaxes(ax, 0.5,remove = [])

fig.savefig('/home/peter/Dropbox/Papers/cancer/IBIN_meeting/ROIs.png',bbox_inches = 'tight',transparent=True,dpi = 300)

fig,ax = plt.subplots()
for i in range(t_courses.shape[0]):
    tc = signal.medfilt(flatten_tc(t_courses[i,:]),5)
    ax.plot(tc - 3*np.arange(t_courses.shape[0])[i],color =colours[i],linewidth = 0.75)
    ax.text(-120, -3*np.arange(t_courses.shape[0])[i],str(i+1),color = colours[i],fontdict = {'fontsize':20}, ha = 'center', va = 'center')
ax.axis('off')

pf.plot_scalebar(ax, 5100, -13, -60*5, 2,thickness = 4)

fig.savefig('/home/peter/Dropbox/Papers/cancer/IBIN_meeting/tcs.png',bbox_inches = 'tight',transparent=True,dpi = 300)
'''
blue = blue/blue.max()
green = green/green.max()

ratio = blue/green 


sz = 256,140

def trim_to_size(stack,sz):
    trim = np.divmod(stack.shape[-2:],sz)[-1]/2    
    return stack[...,int(np.ceil(trim[0])):-int(np.floor(trim[0])),int(np.ceil(trim[1])):-int(np.floor(trim[1]))]

#ratio = trim_to_size(ratio, sz)
ratio_df = (ratio - ratio.mean(0))/ratio.mean(0)

#tifffile.imsave('./ratio_tst_tif.tif',gf.to_16_bit(ratio))
#tifffile.imsave('./green_tst_tif.tif',gf.to_16_bit(green))
#tifffile.imsave('./blue_tst_tif.tif',stack[::2,...])


masks,pts = imf.get_cell_rois(green[0,...],1)

t_course = gf.t_course_from_roi(ratio_df, masks[0])
plt.plot(t_course) 
#pg.image(ratio)

#im = stack[1,...]
def process(stack):
    return signal.medfilt(gf.bin_stack(stack, 1),(1,1,1))


#tst = np.fft.fftshift(np.fft.fft(process(ratio_df),axis = 0),axes = 0)

#tst2 = pywt.wavedec(process(ratio_df),'db7',axis = 0)


#pg.image(np.abs(tst))
'''