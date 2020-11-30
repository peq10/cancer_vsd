#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:57:33 2020

@author: peter
"""
import cancer_functions as canf
from pathlib import Path

topdir = Path('/home/peter/data/Firefly/cancer')
savefile = Path('/home/peter/data/Firefly/cancer/analysis/long_acqs_20201129.csv')
prev_sorted = Path('/home/peter/data/Firefly/cancer/analysis/long_acqs_sorted.csv')
df = canf.get_tif_smr(topdir,savefile,'20201116',None,prev_sorted = prev_sorted,only_long = True)