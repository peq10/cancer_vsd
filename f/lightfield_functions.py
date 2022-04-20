# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:34:33 2018

@author: peq10
"""
import matplotlib.pyplot as plt
import skimage.transform
import scipy.signal
import numpy as np
import time
import scipy.interpolate

#lightfield functions
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff,fs,order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_filter(data, cutoff, fs,btype = 'high', order=5):
    if btype == 'high':
        b, a = butter_highpass(cutoff, fs, order=order)
    elif btype == 'low':
        b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def rotate_test(im,angles):
    ra = []
    for i in angles:
        im_r = skimage.transform.rotate(im,i)
        high_sum_x = butter_filter(np.sum(im_r,0),1,100,btype = 'high')
        ra.append(high_sum_x.max()-high_sum_x.min())
    plt.plot(angles,ra)
    plt.show()
    return angles[np.argmax(ra)]


def get_lf_dims(test_im):
    #find the rotation
    angles =  np.arange(-5,5,0.1)
    im_angle = rotate_test(test_im,angles)
    angles2 = np.arange(im_angle-1,im_angle+1,0.01)
    im_angle = rotate_test(test_im,angles2)
    angles3 = np.arange(im_angle-0.1,im_angle+0.1,0.001)
    im_angle = rotate_test(test_im,angles3)
    
    #now find the image spacing
    im_rotated = skimage.transform.rotate(test_im,im_angle)
    
    sum_x = np.sum(im_rotated,1)
    sum_x_highpass = butter_filter(sum_x,1,100,btype = 'high')
    fftx = np.fft.fft(sum_x_highpass)
    freqs = np.fft.fftfreq(len(sum_x))
    spot_freq = freqs[np.argmax(np.abs(fftx[freqs>0]))]
    spot_period = 1/spot_freq
    print(spot_period)
      
    #now calculate vector
    r = [spot_period*np.sin(im_angle*(np.pi/180)),spot_period*np.cos(im_angle*(np.pi/180))]
    
    #now need to find the center
    center0 = (np.array(test_im.shape)/2).astype(int)
    cent_roi = test_im[center0[0]-int(spot_period):center0[0]+int(spot_period),center0[1]-int(spot_period):center0[1]+int(spot_period)]
    plt.imshow(cent_roi)
    plt.show()
    
    #algo to find a central pixel:
    #find circle location th image that maximises sum.
    #convolve with a cicle or radius r
    radius = 7
    kern = np.zeros((2*radius+1,2*radius+1))
    for idx1 in range(kern.shape[0]):
        for idx2 in range(kern.shape[1]):
            if np.sqrt((idx1-radius)**2+(idx2-radius)**2)<radius:
                kern[idx1,idx2] = 1
            else:
                continue
    conv = scipy.signal.convolve(kern,cent_roi)
    conv = conv[radius:-radius,radius:-radius]
    plt.imshow(conv)
    plt.show()
    coords = np.unravel_index(np.argmax(conv),cent_roi.shape)
    center1 = np.array(coords)+np.array([center0[0]-int(spot_period),center0[1]-int(spot_period)])
    cent_roi = test_im[center1[0]-int(spot_period):center1[0]+int(spot_period),center1[1]-int(spot_period):center1[1]+int(spot_period)]
    plt.imshow(cent_roi)
    plt.show()
    
    #now do a repeat on an interpolation
    x = np.arange(0,2*radius+5,1)
    resampled = scipy.interpolate.RectBivariateSpline(x,x,test_im[(center1[0])-radius-2:center1[0]+radius+3,center1[1]-radius-2:center1[1]+radius+3])
    resampling = 0.01
    x2 = np.arange(0,2*radius+4,resampling)
    upsampled = resampled(x2,x2)
    plt.imshow(upsampled)
    plt.show()
    k0 = (x2*np.ones_like(upsampled)-radius-2)
    k = np.sqrt((k0)**2 + (np.rollaxis(k0,1))**2) < radius
    plt.imshow(k)
    plt.show()
    conv2 = scipy.signal.convolve(k,upsampled)
    plt.imshow(conv2)
    plt.show()
    
    coords2 = np.unravel_index(np.argmax(conv2),conv2.shape)
    offset = (np.array(coords2) -np.array(conv2.shape)/2)*resampling
    
    center2 = center1+offset
    
    return r,center2

def rotate(vec,theta,units = 'degrees'):
    if units == 'degrees':
        theta = theta*np.pi/180
    if vec.shape != (2,1):
        vec = np.matrix(vec).transpose()
    mat = np.matrix([[np.cos(theta),-1*np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.squeeze(np.array(mat*vec))



def get_views_nd_interp(stack,center,r,rad_spots,n_views):
    #also get to work on nonstacks
    if len(stack.shape) == 2:
        stack = np.expand_dims(stack,0)
    
    d = rotate(r,-90)#assuming square array
    #create an array of views
    #viewy, viewx, frame no, y,x
    views = np.zeros((n_views,n_views,stack.shape[0],rad_spots*2+1,rad_spots*2+1))
    
    view_locs = int((n_views-1)/2)
    t0 = time.time()
    ta = time.time()

    interp = scipy.interpolate.RegularGridInterpolator((np.arange(stack.shape[0]),np.arange(stack.shape[1]),np.arange(stack.shape[2])),stack)
    fr = np.arange(stack.shape[0])
    
    t0 = time.time()
    for idx1,view_y in enumerate(range(-1*view_locs,view_locs+1,1)):
        for idx2,view_x in enumerate(range(-1*view_locs,view_locs+1,1)):
            for idx3,px_y in enumerate(np.arange(-rad_spots,rad_spots+1,1)):
                for idx4,px_x in enumerate(np.arange(-rad_spots,rad_spots+1,1)):
                    pos = center+ px_y*d+px_x*r
                    views[idx1,idx2,:,idx3,idx4] = interp((fr,pos[0]+view_y,pos[1]+view_x))
            print(time.time()-t0)

    
    print('total time :'+str(time.time()-ta))
    return views


#define a complex numeric integral fcn  
def complex_quad_integrate(func,a,b,**kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = scipy.integrate.quad(real_func, a, b, **kwargs)
    imag_integral = scipy.integrate.quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])