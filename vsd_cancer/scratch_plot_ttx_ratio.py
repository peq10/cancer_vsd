#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:20:26 2021

@author: peter
"""

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

import seaborn as sns

import scipy.stats as stats

top_dir = Path('/home/peter/data/Firefly/cancer')

data_dir = Path(top_dir,'analysis','full')

df = pd.read_csv(Path(data_dir,'TTX_active_df.csv'))


df['slip_stage'] = df.day.astype(str) + '_'+ df.slip.astype(str) +'_'+ df.stage.astype(str)


uniques = np.unique(df['slip_stage'])


df_sum = pd.DataFrame({'slip_stage':uniques})

num_cells = []
observation_length = []
active_cells = []
neg_active_cells = []
expt = []
stage = []

for data in df_sum.itertuples():
    #aggregate

    rows = df[df.slip_stage == data.slip_stage]
    
    nc = []
    ol = []
    ac = []
    nac = []
    
    for r in rows.itertuples():
        nc.append(r.num_cells)
        ol.append(r.obs_length)
        ac.append(r.total_active)
        nac.append(r.neg_active)
        

    df_sum.loc[data.Index,'num_cells'] = np.sum(nc)
    df_sum.loc[data.Index,'obs_length'] = np.sum(ol)
    df_sum.loc[data.Index,'active_cells'] = np.sum(ac)
    df_sum.loc[data.Index,'neg_active_cells'] = np.sum(nac)
    df_sum.loc[data.Index,'expt'] = r.expts
    df_sum.loc[data.Index,'stage'] = r.stage
    df_sum.loc[data.Index,'Expt_stage'] = r.expts + '_' + r.stage
    df_sum.loc[data.Index,'slip'] = str(r.day) + '_'+ str(r.slip)


df_sum['prop_active'] = df_sum['active_cells']/df_sum['num_cells']
df_sum['neg_prop_active'] = df_sum['neg_active_cells']/df_sum['num_cells']

wash = df_sum[df_sum.expt == 'TTX_10um_washout']
ttx_10 = df_sum[df_sum.expt == 'TTX_10um']
ttx_1 = df_sum[df_sum.expt == 'TTX_1um']

#get the data for ttx10


def get_pairs(df):
    un = np.unique(df.slip)
    unstag = np.unique(df.stage)
    data = np.zeros((len(un),len(unstag)))
    
    order = ['pre','post','washout']
    order2 = ['pre','post']
    if len(unstag) == 3:
        o = np.array([order.index(x) for x in unstag])
    else:
        o = np.array([order2.index(x) for x in unstag])
        
    unstag = unstag[o]
    
    for idx,u in enumerate(un):
        for idx2,u2 in enumerate(unstag):
            data[idx,idx2] = df[((df.slip == u)&(df.stage == u2))].neg_prop_active.values[0]
            
    return data,unstag

wash_data,wash_stages = get_pairs(wash)
t10_data,t10_stages = get_pairs(ttx_10)
t1_data,t1_stages = get_pairs(ttx_1)



fig,ax = plt.subplots()
ax.plot(wash_data.T,'k',linewidth = 2)
ax.plot(wash_data.T,'.k',markersize = 15)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Pre', 'Post', 'Wash'])


fig,ax = plt.subplots()
ax.plot(t10_data.T,'k',linewidth = 2)
ax.plot(t10_data.T,'.k',markersize = 15)
ax.set_xticks([0,1])
ax.set_xticklabels(['Pre', 'Post'])

fig,ax = plt.subplots()
ax.plot(t1_data.T,'k',linewidth = 2)
ax.plot(t1_data.T,'.k',markersize = 15)
ax.set_xticks([0,1])
ax.set_xticklabels(['Pre', 'Post'])