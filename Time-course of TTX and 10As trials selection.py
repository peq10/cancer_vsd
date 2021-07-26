# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 22:41:11 2021

@author: Firefly
"""

import numpy as np
import matplotlib.cm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pyqtgraph as pg

from pathlib import Path

# Exclude hih_k trials/ standard trials/ shorter length trials.
a=pd.read_csv(r'Desktop\long_acqs_20210428_experiments_correct.csv')
a=a.iloc[:,1:]
b=a[(a['use']=='y')&(a['high_k']==False)]

c=b[(b['expt']!='standard')&(b['stage']!='pre')]

d=c[(c['expt']!='standard')&(c['expt']!='TTX_1_pre')&(c['expt']!='TTX_10_pre')&(c['expt']!='TTX_washout_pre')&(c['stage']!='washout')]

active_tcs = []
cell_id = []
names = []
tr = []
n= []
data_folder = Path(Path.home(),r'D:\m+y')
for trial in d['trial_string']:
    trial_string=trial
    trial_save = Path(Path.home(),r'G:\analysis\full\ratio_stacks')
    tc_save = Path(Path.home(),r'D:\all_tcs\all_tcs\y')
    
    results = np.load(Path(trial_save,trial_string,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
    tcs = np.load(Path(data_folder,f'{trial_string}_all_tcs.npy'))
    if tcs.shape[1]==4999:
        names.append(trial_string)
    
        events = results['events'][1]
        ids = [x for x in events.keys() if type(x) != str]
    
        for i in ids:
            active_tcs.append(tcs[i,:])
            cell_id.append(f'{trial_string}_cell_{i}')
            tr.append(trial)
            n.append(i)

active_tcs = np.array(active_tcs)
#np.save(Path(top_dir,'analysis','yilin_tcs.npy'),all_tcs)
#np.save(Path(top_dir,'analysis','yilin_tc_ids.npy'),cell_id)
#np.save(Path(top_dir,'analysis','yilin_names.npy'),names)
active_tcs_filt = ndimage.gaussian_filter(active_tcs,(0,3))
active_tcs_filt=pd.DataFrame(active_tcs_filt)

tr=pd.DataFrame(tr)
tr.columns=['trial_string']
n=pd.DataFrame(n)
n.columns=['id']
notes=pd.concat([tr,n],axis=1)

#active_tcs_filt.to_csv('active_filt_median.csv')
#notes.to_csv('active_notes_median.csv')
