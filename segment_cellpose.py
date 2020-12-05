#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:19:23 2020

@author: peter
"""

import pandas as pd
import numpy as np
from pathlib import Path

from cellpose import models



def segment_cellpose(df_file,save_dir, HPC_num = None):

    df = pd.read_csv(df_file)
    
    ims = []
    savenames = []
    for idx,data in enumerate(df.itertuples()):
        if HPC_num is not None: #allows running in parallel on HPC
            if idx != HPC_num:
                continue
        
        parts = Path(data.tif_file).parts
        trial_string = '_'.join(parts[parts.index('cancer'):-1])
        trial_save = Path(save_dir,'ratio_stacks',trial_string)
        
        ims.append(np.load(Path(trial_save,f'{trial_string}_im.npy')))
        savenames.append(Path(trial_save,f'{trial_string}_seg.npy'))
        
        
    model = models.Cellpose(gpu=False, model_type='cyto')
    masks, flows, styles, diams = model.eval(ims, diameter=30, channels=[0,0])
    
    
    for idx in range(len(ims)):
        np.save(savenames[idx],masks[idx])