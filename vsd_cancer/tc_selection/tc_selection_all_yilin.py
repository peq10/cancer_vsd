# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 16:50:29 2021

@author: Firefly
"""

import numpy as np
import matplotlib.cm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pyqtgraph as pg

from pathlib import Path

criteria='new'


initial_df=pd.read_csv(r'G:\analysis\long_acqs_20210428_experiments_correct.csv')
t=initial_df[initial_df['use']=='y']

active_tcs = []
cell_id = []
names = []
tr = []
n= []
#data_folder = Path(Path.home(),r'D:\m+y')
for trial in t['trial_string']:
    trial_string=trial
    trial_save = Path(Path.home(),r'G:\analysis\full\ratio_stacks')
    tc_save = Path(Path.home(),r'D:\all_tcs\all_tcs\y')
    
    results = np.load(Path(trial_save,trial_string,f'{trial_string}_event_properties_yilin_copy_{criteria}_criteria.npy'),allow_pickle = True).item()
    #tcs = np.load(Path(data_folder,f'{trial_string}_all_tcs.npy'))
    tcs=np.load(Path(trial_save,trial_string,f'{trial_string}_all_eroded_median_tcs.npy'))
    
    if tcs.shape[1]==4999:
 
        names.append(trial_string)
    
        events = results['events'][1]
        ids = [x for x in events.keys() if type(x) != str]
    
        for i in ids:
            active_tcs.append(tcs[i,:])
            cell_id.append(f'{trial_string}_cell_{i}')
            tr.append(trial)
            n.append(i)

tr=pd.DataFrame(tr)
tr.columns=['trial_string']
n=pd.DataFrame(n)
n.columns=['id']
cell_labels=pd.concat([tr,n],axis=1)
active_tcs = np.array(active_tcs)
#np.save(Path(top_dir,'analysis','yilin_tcs.npy'),all_tcs)
#np.save(Path(top_dir,'analysis','yilin_tc_ids.npy'),cell_id)
#np.save(Path(top_dir,'analysis','yilin_names.npy'),names)
active_tcs_filt = ndimage.gaussian_filter(active_tcs,(0,3))
cell_labels.to_csv(f'cell_labels_yilin.csv')
