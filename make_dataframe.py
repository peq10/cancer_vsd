#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:57:33 2020

@author: peter
"""
import cancer_functions as canf
from pathlib import Path
import datetime

home = Path.home()

if 'peq10' in str(home):
    HPC = True
    top_dir = Path(home,'firefly_link/cancer')
    savestr = '_HPC'
else:
    HPC = False
    top_dir = Path(home,'data/Firefly/cancer')
    savestr = ''


save_file = Path(top_dir,'analysis',f'long_acqs_{datetime.datetime.now().year}{datetime.datetime.now().month:02}{datetime.datetime.now().day:02}{savestr}.csv')
prev_sorted = Path(top_dir,'analysis','long_acqs_20201129_sorted.csv')
    

df = canf.get_tif_smr(top_dir,save_file,'20201116',None,prev_sorted = prev_sorted,only_long = True)
