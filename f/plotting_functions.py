# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:37:35 2018

@author: peq10
"""
import numpy as np
import matplotlib.colors
import scipy.ndimage as ndimage
import matplotlib.cm

def hex2rgb(hex_string):
    return np.array([int(hex_string[1+2*i:1+2*(i+1)],16) for i in range(3)]).astype(np.uint8)

def make_colormap_rois(masks,cmap):
    if type(cmap) == str:
        cmap = matplotlib.cm.__dict__[cmap]
    outlines = np.array([np.logical_xor(mask,ndimage.morphology.binary_dilation(mask)) for mask in masks]).astype(int)
    outlines *= np.arange(outlines.shape[0]+1)[1:,None,None] 
    outlines = np.ma.masked_less(outlines,1)
    overlay = cmap(np.sum(outlines,0)/outlines.shape[0])
    return overlay, cmap(np.arange(outlines.shape[0]+1)/outlines.shape[0])[1:]


def label_roi_centroids(ax,masks,colours,fontdict = None):
    coms = [ndimage.center_of_mass(mask) for mask in masks]
    for idx,com in enumerate(coms):
        ax.text(com[1],com[0],str(idx),color = colours[idx],fontdict = fontdict, ha = 'center', va = 'center')


def get_bar_lengths_for_normalised(vector,length_required_not_normalised):
    vector_range = np.max(vector)-np.min(vector)
    return length_required_not_normalised/vector_range


def set_thickaxes(ax,thickness,remove = ['top','right']):
    
    if remove == 'all':
        remove = ['top','right','left','bottom']

    for key in remove:
        ax.spines[key].set_visible(False)
    
    for key in ax.spines:
        if key in remove:
            continue
        else:
            ax.spines[key].set_linewidth(thickness)

    ax.xaxis.set_tick_params(width=thickness,direction = 'out')

    ax.yaxis.set_tick_params(width=thickness,direction = 'out')

def set_minortick_width(ax,width):
    ax.tick_params(which = 'minor',width = width)
    
def set_all_fontsize(ax,fontsize):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
        
        
def plot_scalebar(ax,x_pos,y_pos,x_len,y_len,thickness = 2,color = 'k'):
    ax.plot([x_pos,x_pos],[y_pos,y_pos+y_len],linewidth = thickness,color = color)
    ax.plot([x_pos,x_pos+x_len],[y_pos,y_pos],linewidth = thickness,color = color)
    

def cust_colormap():
    cdict = {'red':   ((0.0,  1.0, 1.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}
    return matplotlib.colors.LinearSegmentedColormap('cust', cdict)

def cust_colormap_cust(red = 0,green = 1,blue = 0):
    cdict = {'red':   ((0.0,  red, red),
                   (1.0,  red, red)),

         'green': ((0.0,  green, green),
                   (1.0,  green, green)),

         'blue':  ((0.0,  blue, blue),
                   (1.0,  blue, blue))}
    return matplotlib.colors.LinearSegmentedColormap('cust', cdict)

         
def make_square_plot(ax):
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()),adjustable = 'box')

def set_aspect_ratio_loglog(ax, aspect_ratio):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_aspect(aspect_ratio * ((np.log10(x_max / x_min)) / (np.log10(y_max / y_min))))
        
def set_aspect_ratio_semilog(ax, aspect_ratio,axis,ticks):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    if axis == 'x':
        ax.set_aspect(aspect_ratio * ((np.log10(x_max / x_min)) / ((y_max / y_min))))
    else:
        ax.set_aspect(aspect_ratio * (((x_max / x_min)) / (np.log10(y_max / y_min))))
        
        
def add_significance_bar(ax,value,x_points,y_points,textLoc = None,textFormat = None,fontdict = None):
    if textLoc is None:
        x,y = (x_points[1]+x_points[0])/2,y_points[1]+1.8
    else:
        x,y = (x_points[1]+x_points[0])/2,textLoc
        
    if fontdict is None:
        fontdict = {'size':15}
        
    ax.plot([x_points[0],x_points[1]],[y_points[1],y_points[1]],color = 'k',linewidth = 3)
    ax.plot([x_points[1],x_points[1]],[y_points[1],y_points[0]],color = 'k',linewidth = 3)
    ax.plot([x_points[0],x_points[0]],[y_points[1],y_points[0]],color = 'k',linewidth = 3)
    if 'n' not in value:
        if textFormat is None:
            ax.text(x,y,'p = %.E'%(value),ha = 'center',fontdict = fontdict)
        else:
            ax.text(x,y,('p = %.'+str(textFormat)+'f')%(np.round(value,decimals = textFormat)),ha = 'center',fontdict = fontdict)
    else:
        ax.text(x,y,value,ha = 'center',fontdict = fontdict)
        
        
def alpha_composite(top_image,bottom_image):
    '''
    Requires uint8

    '''
    RGB1 = top_image[...,:3]
    RGB2 = bottom_image[...,:3]
    
    a1 = top_image[...,3]/255.0
    a2 = bottom_image[...,3]/255.0
    
    a3 = a1 + a2*(1-a1)
    RGB3 = RGB1*a1[...,None] + RGB2*a2[...,None]*(1-a1[...,None])/a3[...,None]
    
    return np.round(np.concatenate((RGB3,a3[...,None]*255),axis = -1)).astype(np.uint8)