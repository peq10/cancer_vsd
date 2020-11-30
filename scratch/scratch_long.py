#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:55:44 2020

@author: peter
"""

#a scratch process long im script

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import scipy.ndimage as ndimage
from pathlib import Path
import pandas as pd
import h5py


import cancer_functions as canf


topdir = Path('/home/peter/data/Firefly/cancer')

savefile = Path(topdir,'analysis','long_acqs_sorted.csv')


df = pd.read_csv(savefile)
failed = []

for data in df.itertuples():

    
    
    parts = Path(data.tif_file).parts
    trial_string = '_'.join(parts[parts.index('cancer'):-1])
    
    try:
        result_dict = canf.load_and_slice_long_ratio(data.tif_file,
                                                     data.SMR_file,
                                                     T_approx = 3*10**-3,
                                                     fs = 5)
    except Exception as err:
        print(trial_string)
        print(err)
        failed.append([trial_string,err])
        continue

        

    save_dir = Path(topdir,'analysis/ratio_stacks',trial_string)
    if not save_dir.is_dir():
        save_dir.mkdir(parents = True)
    
    for key in result_dict.keys():
        np.save(Path(save_dir,f'{trial_string}_{key}.npy'),result_dict[key])
    
    print(f'Saved {trial_string}')
    
    rat = np.copy(result_dict['ratio_stack'])
    im = result_dict['im']
    del result_dict
    
    rat2 = ndimage.gaussian_filter(rat,(2,1,1))
    std_im = np.std(rat2,0)
    
    fig,axarr = plt.subplots(ncols = 2)
    axarr[0].imshow(im,cmap = 'Greys_r')
    axarr[1].imshow(std_im)
    axarr[0].axis('off')
    axarr[1].axis('off')
    fig.savefig(Path(topdir,'analysis/ims',trial_string+'.png'),bbox_inches = 'tight',dpi = 300)
    
    del rat
    del rat2
    