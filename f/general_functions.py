# -*- coding: utf-8 -*-
"""
A set of general usage functions to complement IAF

Created on Thu Jul  5 10:38:52 2018

@author: peq10
"""
import numpy as np
import scipy
import tifffile
import glob
import scipy.ndimage as ndimage
import scipy.constants
import os
import scipy.stats
import psutil
import numpy.ma as ma
import scipy.signal
#import read_roi
import matplotlib.cm as cm
import matplotlib.colors
import skimage.draw

if __name__ =='__main__':
    import read_roi_cust as rr
else:
    from . import read_roi_cust as rr
    
    
def to_df(stack,offset = 0):
    slopes,intercept,_ = stack_linregress(stack)
    bck = slopes*np.arange(stack.shape[0])[:,None,None] + intercept
    return 100*(stack-bck)/(bck - offset),slopes,intercept
    
def depth_code_arr(arr, cmap = None, normalise_depths = True):
    '''
    Takes a 3D array and returns a 2D image which is a max projection of 
    the 3D array along the first axis with pixels color coded by depth.
    
    '''
    #assumes z,x,y
    if cmap is None:
        cmap = cm.viridis
    depths = np.linspace(0,1,len(arr))
    colors = cmap(depths)[:,:-1]
    colors_hsv = matplotlib.colors.rgb_to_hsv(colors)
    #replace v with image value for each depth
    
    if normalise_depths:
        arr = arr/arr.max(-1).max(-1)[:,None,None]
    else:
        arr = arr/arr.max()
        
    #set value for each depth
    colored_arr = arr*colors_hsv[:,None,None,-1]
    #make colors for each depth
    colored_arr = np.concatenate((np.ones_like(colored_arr[:,:,:,None])*colors_hsv[:,None,None,:-1],colored_arr[...,None]), axis = -1)
    
    colored_arr = matplotlib.colors.hsv_to_rgb(colored_arr)
    
    #calculate max pixel depth for each
    px_depth = np.argmax(np.sum(colored_arr**2,-1),0)
    
    #and select the correct elements
    z_coded = np.take_along_axis(colored_arr,px_depth[None,:,:,None],axis = 0)[0,...]
    
    return z_coded


def datafilterbn( raw_trace, num_passes ):
  filterdata1 = raw_trace.copy()
  filterdata2 = filterdata1

  for i in range( num_passes ):
    for j in range( 1, len( filterdata1 )-1 ):
      filterdata2[j] = (filterdata1[j-1] + (2*filterdata1[j]) + filterdata1[j+1] ) / 4
    filterdata2[0] = filterdata2[1]
    filterdata2[ -1 ] = filterdata2[ -2 ]
    filterdata1 = filterdata2

  return filterdata2


def upsample_image(im,factor):
    s = im.shape
    return np.reshape(im[:,None,:,None]*np.ones((s[0],factor,s[1],factor)),(s[0]*factor,s[1]*factor))

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))


def parse_datetime_metadata(string):
    date = string[:4]+string[5:7]+string[8:10]
    time = string[11:13]+string[14:16]+string[17:19]
    return date,time

def to_16_bit(array):
    return (norm(array)*(2**16-1)).astype(np.uint16)

def to_8_bit(array):
    return (norm(array)*(2**8-1)).astype(np.uint8)

def t_course_from_roi(nd_stack,roi):
    masked = np.ma.masked_less(roi.astype(int),1)
    if len(roi.shape) == 2:
        return np.mean(np.mean(nd_stack*masked[None,...],-1),-1).data
    else:
        sh1 = nd_stack.shape #assume [...,t,y,x]
        sh2 = roi.shape #assume [...,y,x] and want to keep all dims
        i = len(sh1) - 2
        j = len(sh2) - 2
        nd_stack = nd_stack.reshape(sh1[:-3] + tuple(np.ones(j,dtype = int))+sh1[-3:])
        masked = masked.reshape(sh2[:-2]+tuple(np.ones(i,dtype = int))+sh2[-2:])
        return np.mean(np.mean(nd_stack*masked,-1),-1).data
        

def trailing_dim_mean(stack):
    return stack.reshape((stack.shape[0],-1)).mean(axis = 1)

def mem_check(max_mem_percentage = 95,response = 'throw_error'):
    #a check when loading things, etc. to try and stop freezing
    if psutil.virtual_memory().percent > max_mem_percentage:
        if response == 'throw_error':
            raise MemoryError
        else:
            return None
    else:
        return None

def read_redshirt_images(file_path):
    data = np.fromfile(file_path,dtype = np.int16)
    header_size = 2560
    header = data[:header_size]
    ncols, nrows = map(int, header[384:386])  # prevent int16 overflow
    nframes = int(header[4])
    frame_interval = header[388] / 1000
    if frame_interval >= 10:
        frame_interval *= header[390]  # dividing factor
    image_size = nrows * ncols * nframes
    bnc_start = header_size + image_size
    images = np.reshape(np.array(data[header_size:bnc_start]),
                        (nrows, ncols, nframes))
    return np.rollaxis(images,-1)

def check_and_make_directory(file_path):
     directory = os.path.dirname(file_path)
     if not os.path.isdir(directory):
         os.makedirs(directory)

def memory_efficient_average_stack(path,key,omit_fifth = True,keep_label = None,limit_repeats = False,num_repeats = None):
    filenames = getFilenamesByExtension(path)
    #excludes some of the stacks
    #get all file numbers
    nums = []
    for filename in filenames:
        loc = filename.find(key)+len(key)
        num = get_int_from_string(filename,loc)
        nums.append(num)


    if keep_label is not None:
        filenames = [f for idx,f in enumerate(filenames) if nums[idx] in keep_label]
        nums = [n for n in nums if n in keep_label]

    if omit_fifth:
        #find first non omitted stack
        first_loc = np.where(np.array(nums)%5 !=0)[0][0]
        mean_stack = tifffile.imread(filenames[first_loc]).astype(np.float64)
        #now load and average stacks
        count = 1
        for idx,filename in enumerate(filenames):
            if idx == first_loc or idx in np.where(np.array(nums)%5 ==0)[0] :
                continue
            print(count)
            if count ==num_repeats and limit_repeats:
                break
            mean_stack += tifffile.imread(filename).astype(np.float64)
            count += 1


        mean_stack = mean_stack/len(np.where(np.array(nums)%5 !=0)[0])
        return mean_stack
    else:
        mean_stack = tifffile.imread(filenames[0]).astype(np.float64)
        count = 1
        for idx,filename in enumerate(filenames[1:]):
            if count ==num_repeats and limit_repeats:
                break
            mean_stack += tifffile.imread(filename).astype(np.float64)
            count += 1

        return mean_stack/(idx+1)



def getFilenamesByExtension(path,fileExtension = '.tif',recursive_bool = True):
    if recursive_bool:
        return [file for file in glob.glob(path + '/**/*'+fileExtension, recursive=recursive_bool)]
    else:
        return [file for file in glob.glob(path + '/**'+fileExtension, recursive=recursive_bool)]



def sort_zipped_lists(list_of_lists,key_position = 0):
    res = zip(*sorted(zip(*list_of_lists),key = lambda x:x[key_position]))
    return [list(i) for i in res]

def get_int_from_string(string,loc,direction = 1):
    count = 0
    while True:
        try:
            if direction == 1:
                int(string[loc:loc + count +1])
                if loc +count > len(string):
                    break
            elif direction == -1:
                int(string[loc-count:loc+1])
            else:
                raise ValueError('Direction argument must be 1 or -1')
            count += 1
        except Exception:
            break

    if direction == 1:
        return int(string[loc:loc + count])
    elif direction == -1:
        return int(string[loc-count+1:loc+1])


def load_repeats_and_sort(path,key,omit_fifth = True):
    filenames = getFilenamesByExtension(path)

    if omit_fifth:
        stacks = []
        nums = []
        back_stacks = []
        for idx,filename in enumerate(filenames):
            stack = tifffile.imread(filename).astype(np.float64)
            loc = filename.find(key)+len(key)
            num = get_int_from_string(filename,loc)
            if num%5 != 0:
                nums.append(num)
                stacks.append(stack)
            else:
                back_stacks.append(stack)

        sorted_stacks = sort_zipped_lists([nums,stacks])
        return sorted_stacks[1],back_stacks
    else:
        stacks = []
        nums = []
        for idx,filename in enumerate(filenames):
            stack = tifffile.imread(filename).astype(np.float64)
            loc = filename.find(key)+len(key)
            num = get_int_from_string(filename,loc)
            nums.append(num)
            stacks.append(stack)


        sorted_stacks = sort_zipped_lists([nums,stacks])
        return sorted_stacks[1]


def correlationMap(stack):
    #a function that plots themean correlation of a pixel with its neighbourse
    stack = stack/np.sum(stack,0)
    correlationMap = np.zeros_like(stack[0,:,:])
    for idx1 in range(stack.shape[1]-1):
        if idx1 == 0:
            continue
        for idx2 in range(stack.shape[2]-1):
            if idx2 == 0:
                continue
            correlations = []
            for neighbour1 in [-1,0,1]:
                for neighbour2 in [-1,0,1]:
                    #measure neighbour correlation
                    correlations.append(scipy.stats.pearsonr(stack[:,idx1,idx2],stack[:,idx1+neighbour1,idx2+neighbour2])[0])
            correlationMap[idx1,idx2] = (np.sum(correlations)-1)/(8)
    return correlationMap

def stack_covariance(vec,vec2):
    return np.cov(vec,vec2,bias = 1).flat

def stack_linregress(stack,vec = None):
    '''
    Does the vectorised linear regression of scipy.stats.linregress(np.arange(stack.shape[0]),stack) along 0th dim.
    '''
    if vec is None:
        vec = np.arange(stack.shape[0])

    cov_coeff = np.apply_along_axis(stack_covariance,0,stack,vec)
    ssxym = cov_coeff[2,...]
    ssxm = cov_coeff[3,...]
    ssym = cov_coeff[0,...]
    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)
    r = r_num / r_den
    r[r>1] = 1
    r[r<-1] = -1
    r[np.isnan(r)] = 0
    slope = r_num / ssxm
    ymean = np.mean(stack,0)
    xmean = np.mean(np.arange(stack.shape[0]))
    intercept = ymean - slope*xmean
    return slope,intercept,r


def detrend_pixels(stack,fit_type = 'linear'):
    #a function to subtract a linear trend from each pixel in the stack
    detrended = np.zeros_like(stack)
    trends = np.zeros_like(stack)
    x = np.arange(stack.shape[0])
    fits = np.zeros((2,stack[0,:,:].shape[0],stack[0,:,:].shape[1]))
    for idx1 in range(stack.shape[1]):
        for idx2 in range(stack.shape[2]):
            if fit_type == 'linear':
                fit = scipy.stats.linregress(x,stack[:,idx1,idx2])
                fit_eval = fit.slope*x +fit.intercept
            elif fit_type == 'exp':
                fit = scipy.stats.linregress(x,np.log(stack[:,idx1,idx2]))
                fit_eval = np.exp(fit.slope*x +fit.intercept)
            else:
                raise ValueError('fit type not recognised')

            detrended[:,idx1,idx2] = stack[:,idx1,idx2] - fit_eval
            trends[:,idx1,idx2] = fit_eval
            fits[0,idx1,idx2] = fit.slope
            fits[1,idx1,idx2] = fit.intercept

    return detrended,trends,fits

def filter_pixels(stack,filter_type = 'median',kernel_size = [3,1,1]):

    if filter_type == 'median':
        return scipy.signal.medfilt(stack,kernel_size = [kernel_size,0,0])
    else:
        raise ValueError('Filter type not recognised')


def grouped_Z_project(stack,groupSize,projectType = 'mean'):
    #does a grouped z project like in imagej
    #trim stack if groupSize doesnt fit
    remainder  = stack.shape[0]%groupSize
    if remainder !=0:
        stack = stack[:-remainder,:,:]

    groupedStack = np.zeros((int(stack.shape[0]/groupSize),stack.shape[1],stack.shape[2])).astype(stack.dtype)

    for idx in range(groupedStack.shape[0]):
        if projectType == 'mean':
            groupedStack[idx,:,:] = np.mean(stack[idx*groupSize:idx*groupSize+groupSize+1,:,:],0)
        elif projectType == 'max':
            groupedStack[idx,:,:] = np.max(stack[idx*groupSize:idx*groupSize+groupSize+1,:,:],0)
        elif projectType == 'sum':
            groupedStack[idx,:,:] = np.sum(stack[idx*groupSize:idx*groupSize+groupSize+1,:,:],0)
        else:
            raise ValueError('Project type not recognised')
        #replace zeros in frame with mean of surrounding pixels
        if (groupedStack[idx,:,:] == 0).all():
            raise ValueError('Image is all zeros')



    return groupedStack

def two_photon_res(wl,NA):
    return (0.383*wl)/NA**0.91

def power_to_photon_flux(wl,power,NA = 0.8):
    spot_area = 2*np.pi*two_photon_res(wl,NA)**2
    photon_flux = power*(wl/(scipy.constants.h*scipy.constants.c))
    flux_density = photon_flux/spot_area
    return flux_density

def norm(array):
    return (array - np.min(array))/(np.max(array) - np.min(array))

def read_roi_file(roi_filepath,im_dims = None):
    with open(roi_filepath,'rb') as f:
        roi = rr.read_roi(f)

    if im_dims is not None:
        return roi, skimage.draw.polygon2mask(im_dims,roi)
    else:
        return roi

def radial_average_profile(array,center):
    #from http://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((array.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), array.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def radial_max_profile(array,center):
    #from http://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((array.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)

    #get the max for each distance
    raise ValueError('Not implemented')
    tbin = np.bincount(r.ravel(), array.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


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

def get_tif_metadata(tif_filepath):
    raise ValueError('Deprecated! fix')
    with tifffile.TiffFile(tif_filepath) as tif:
        meta_dict = tif.micromanager_metadata
    return meta_dict


def array_from_radial_profile(radial_average,array_shape,center = None):
    array3 = np.zeros(array_shape).astype(radial_average.dtype)
    y, x = np.indices(array_shape)
    if center is None:
        center = np.array(array_shape)/2
    r = np.round(np.sqrt((x - center[0])**2 + (y - center[1])**2)).astype(int)
    ravelled_r = np.ravel(r)
    if ravelled_r.max()>len(radial_average)-1:
        radial_average = np.pad(radial_average,(0,ravelled_r.max()-len(radial_average)+1),'edge')
    array = radial_average[ravelled_r]
    coords = np.unravel_index([i for i in range(len(array))],array_shape)
    array3[coords[0],coords[1]] = array
    return array3

def array_from_radial_profile_oversamp(radial_average,array_shape,oversampling = 2,center = None):
    #allows higher radial sampling to reduce quantisation artefacts
    y, x = np.indices(array_shape)
    if center is None:
        center = np.array(array_shape)/2
    r = np.round(oversampling*np.sqrt((x - center[0])**2 + (y - center[1])**2)).astype(int)
    ravelled_r = np.ravel(r)
    if ravelled_r.max()>len(radial_average)-1:
        radial_average = np.pad(radial_average,(0,ravelled_r.max()-len(radial_average)+1),'edge')
    array = radial_average[ravelled_r]
    print('THIS FUNCTION DOESNT WORK!')
    return np.reshape(array,array_shape)

def bin_stack(stack,bin_size):
    #takes x-y bin of stack, image

    #pad leading edge to make the shape correct
    remainders = np.mod(stack.shape,bin_size)
    if remainders[1] !=0:
        stack = np.pad(stack,((0,0),(remainders[0],0),(0,0)),'edge')
    if remainders[2] != 0:
        stack = np.pad(stack,((0,0),(0,0),(remainders[1],0)),'edge')

    new_shape =(stack.shape[0],)+tuple(((np.array(stack.shape[1:])/bin_size)).astype(int))
    shape = (new_shape[0], new_shape[1], stack.shape[1] // new_shape[1], new_shape[2], stack.shape[2] // new_shape[2])
    return stack.reshape(shape).mean(-1).mean(-2)


def bin_array(array,bin_size):
    new_shape =((np.array(array.shape)/bin_size)).astype(int)
    shape = (new_shape[0], array.shape[0] // new_shape[0], new_shape[1], array.shape[1] // new_shape[1])
    return array.reshape(shape).mean(-1).mean(1)
