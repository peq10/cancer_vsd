#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:32:43 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.gridspec as gridspec

import pandas as pd

from pathlib import Path

import f.plotting_functions as pf

import scipy.stats as stats

import astropy.stats as ass


from vsd_cancer.functions import stats_functions as statsf


top_dir = Path('/home/peter/data/Firefly/cancer')

data_dir = Path(top_dir,'analysis','full')

df = pd.read_csv(Path(data_dir,'TTX_active_df_by_cell.csv'))



#df = df[df.expt == 'standard']

T = 0.2


df['exp_stage'] = df.expt + '_' + df.stage

df['event_rate'] = (df['n_neg_events'] +  df['n_pos_events'])/(df['obs_length']*T)
df['neg_event_rate'] = (df['n_neg_events'] )/(df['obs_length']*T)

df['integ_rate'] = (df['integrated_events'])/(df['obs_length']*T)
df['neg_integ_rate'] = (df['neg_integrated_events'] )/(df['obs_length']*T)


use = [x for x in np.unique(df['exp_stage']) if 'washout' not in x]


dfn = df.copy()
#dfn = dfn[dfn.event_rate < 0.03]
    
    
use_bool = np.array([np.any(x in use) for x in dfn.exp_stage])
dfn = dfn[use_bool]

nbins = 20
key = 'neg_event_rate'
log = True
histtype = 'step'
bins = np.histogram(dfn[key],bins = nbins)[1]


pre_10 = dfn[dfn.exp_stage == 'TTX_10um_pre'][key]
post_10 = dfn[dfn.exp_stage == 'TTX_10um_post'][key]
pre_1 = dfn[dfn.exp_stage == 'TTX_1um_pre'][key]
post_1 = dfn[dfn.exp_stage == 'TTX_1um_post'][key]









def plot_average_cis_1_10(pre_10,post_10,pre_1,post_1, function = np.mean,num_resamplings = 10**5,scale = 4):
    
    CI_pre_10,pre_10_resamplings = statsf.construct_CI(pre_10,5, num_resamplings = num_resamplings)
    CI_post_10,post_10_resamplings = statsf.construct_CI(post_10, 5, num_resamplings = num_resamplings)
    CI_pre_1,pre_1_resamplings = statsf.construct_CI(pre_1,5, num_resamplings = num_resamplings)
    CI_post_1,post_1_resamplings = statsf.construct_CI(post_1, 5, num_resamplings = num_resamplings)

    
    
    p_10 = statsf.bootstrap_test(pre_10,post_10,function = function,plot = False,num_resamplings = num_resamplings)
    p_1 = statsf.bootstrap_test(pre_1,post_1,function = function,plot = False,num_resamplings = num_resamplings)

    
    #TODO print n cells etc. to a file
    print(f'10 um: {p_10[0]}, 1 um: {p_1[0]}')
    vals = np.array([np.mean(pre_10),np.mean(post_10),np.mean(pre_1),np.mean(post_1)])*10**scale
    errors = np.array([CI_pre_10,CI_post_10,CI_pre_1,CI_post_1])*10**scale
    
    
    fig,ax = plt.subplots()
    pf.plot_errorbar(ax,vals[:2],errors[:2,:])
    pf.plot_errorbar(ax,vals[2:],errors[2:,:],off = 2)
    sc_str = '10$^{-'+str(scale)+'}$'
    ax.set_ylabel('Negative Event Rate\nper Cell (x'+sc_str+' s$^{-1}$)')
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(['Pre', 'Post 10 uM', 'Pre', 'Post 1 uM'])
    
    
    pf.add_significance_bar(ax, p_10[0], [0,1], np.array([1.075,1.1])*errors.max(), textLoc = 1.15*errors.max())
    pf.add_significance_bar(ax, p_1[0], [2,3], np.array([1.075,1.1])*errors.max(), textLoc = 1.15*errors.max())

    pf.set_all_fontsize(ax, 16)
    pf.set_thickaxes(ax, 3)
    pf.make_square_plot(ax)
    
    return fig
    

def plot_TTX_summary(df,use,key = 'event_rate', function = np.mean):
    dfn = df.copy()     
        
    use_bool = np.array([np.any(x in use) for x in dfn.exp_stage])
    dfn = dfn[use_bool]
    
    pre_10 = dfn[dfn.exp_stage == 'TTX_10um_pre'][key].to_numpy()
    post_10 = dfn[dfn.exp_stage == 'TTX_10um_post'][key].to_numpy()
    pre_1 = dfn[dfn.exp_stage == 'TTX_1um_pre'][key].to_numpy()
    post_1 = dfn[dfn.exp_stage == 'TTX_1um_post'][key].to_numpy()
    
    fig1 = plot_average_cis_1_10(pre_10,post_10,pre_1,post_1)
    
    
    return fig1
        

plot_TTX_summary(df,use)



#plt.figure()
#plt.hist(pre_resamplings)
#plt.hist(post_resamplings)
#plt.hist(wash_resamplings)
#now do bootstrapping
#bootstrap_null = ass.bootstrap(pre,bootnum = 10000,samples = None, bootfunc = np.sum) 