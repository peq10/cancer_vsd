#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 10:15:14 2021

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

import seaborn as sns


from vsd_cancer.functions import stats_functions as statsf


top_dir = Path('/home/peter/data/Firefly/cancer')

data_dir = Path(top_dir,'analysis','full')

df = pd.read_csv(Path(data_dir,'non_ttx_active_df_by_cell.csv'))



#df = df[df.expt == 'standard']

T = 0.2


df['exp_stage'] = df.expt + '_' + df.stage
df['day_slip'] = df.day.astype(str) + '_' + df.slip.astype(str) 


df['event_rate'] = (df['n_neg_events'] +  df['n_pos_events'])/(df['obs_length']*T)
df['neg_event_rate'] = (df['n_neg_events'] )/(df['obs_length']*T)

df['integ_rate'] = (df['integrated_events'])/(df['obs_length']*T)
df['neg_integ_rate'] = -1*(df['neg_integrated_events'] )/(df['obs_length']*T)

def f(arr):
    return np.percentile(95,arr)

def plot_average_mda_mcf(mda,mcf,tgf, function = np.mean,num_resamplings = 10**5, scale = 3, density = True):
  
   
    
    #p_mda_mcf = statsf.bootstrap_test(mda,mcf,function = function,plot = True,num_resamplings = num_resamplings, names = ['MDA-MB-231', 'MCF10A'])
    #p_mda_tgf = statsf.bootstrap_test(mda,tgf,function = function,plot = True,num_resamplings = num_resamplings, names = ['MDA-MB-231', 'MCF10A + TGF$\\beta$'])
    #p_mcf_tgf = statsf.bootstrap_test(tgf,mcf,function = function,plot = True,num_resamplings = num_resamplings, names = ['MCF10A + TGF$\\beta$', 'MCF10A'])
    
    
    #TODO print n cells etc. to a file
    #print(f'mda-mcf: {p_mda_mcf[0]}, mda_tgf: {p_mda_tgf[0]}, mcf-tgf: {p_mcf_tgf[0]}')
    
    bins = np.histogram(np.concatenate((mda,mcf,tgf))*10**3,bins = 20)[1]
    

    fig,axarr = plt.subplots(nrows = 3)
    c = 0.05
    h0 = axarr[0].hist(mda*10**scale,bins = bins, log = True, density = density, label = 'MDA-MB-231', color = (c,c,c))
    h1 = axarr[1].hist(mcf*10**scale,bins = bins, log = True,  density = density, label = 'MCF10A', color = (c,c,c))
    h2 = axarr[2].hist(tgf*10**scale,bins = bins, log = True,  density = density, label = 'MCF10A+TGF$\\beta$', color = (c,c,c))
    
    ma = np.max([h0[0].max(),h1[0].max(),h2[0].max()])
    
    for idx,a in enumerate(axarr):
        if not density:
            a.set_ylim([0.6,10**4.5])
            a.set_yticks(10**np.arange(0,4,3))
        a.legend(frameon = False,loc = (0.4,0.4),fontsize = 16)
        pf.set_all_fontsize(a, 16)
        if idx != 2:
            a.set_xticklabels([])
            
        
    if not density:
        axarr[1].set_ylabel('Number of cells')
    else:
        axarr[1].set_ylabel('Propotion of cells')

    axarr[-1].set_xlabel(f'Integrated event rate per {10**scale} cells ' + '(%$\cdot$s / s)')

    
    
    #pf.add_significance_bar(ax, p_mda_mcf[0], [0,0.975], np.array([1.05,1.075])*errors.max(), textLoc = 1.1*errors.max())
    #pf.add_significance_bar(ax, p_mcf_tgf[0], [1.025,2], np.array([1.05,1.075])*errors.max(), textLoc = 1.1*errors.max())
    #pf.add_significance_bar(ax, p_mda_tgf[0], [0,2], np.array([1.2,1.225])*errors.max(), textLoc = 1.25*errors.max())
    
    #pf.set_all_fontsize(ax, 16)
    #pf.set_thickaxes(ax, 3)
    #pf.make_square_plot(ax)
    return fig
    
    
def plot_MCF_summary(df,key = 'neg_integ_rate', function = np.mean):
    dfn = df.copy()

    mda = dfn[dfn.exp_stage == 'standard_none'][[key, 'day_slip', 'cell']]
    mcf = dfn[dfn.exp_stage == 'MCF10A_none'][[key, 'day_slip', 'cell']]
    tgf = dfn[dfn.exp_stage == 'MCF10A_TGFB_none'][[key, 'day_slip', 'cell']]

    md = mda[key].to_numpy()
    mc = mcf[key].to_numpy()
    tg = tgf[key].to_numpy()

    tst1 = mda.groupby('day_slip').mean()[key].to_numpy()
    tst2 = mcf.groupby('day_slip').mean()[key].to_numpy()
    tst3 = tgf.groupby('day_slip').mean()[key].to_numpy()


    fig1 = plot_average_mda_mcf(mda[key].to_numpy(),mcf[key].to_numpy(),tgf[key].to_numpy(),function = function)
    ax = fig1.gca()
    #ax.plot(np.ones(len(md))-1,md*10**4,'.')
    #ax.plot(np.ones(len(mc))-0,mc*10**4,'.')
    #ax.plot(np.ones(len(tg))+1,tg*10**4,'.')
    #plt.axis('auto')
    
    scale = 4
    
    bins = np.histogram(np.concatenate((md,mc,tg))*10**scale,bins = 10)[1]
    
    log = True
    t = 'step'
    cum = False
    density = True
    fig,ax  = plt.subplots()
    mdh = ax.hist(md*10**scale,bins = bins,density = density, histtype = t, cumulative = cum, log = log, label = 'MDA-MB-231',linewidth = 3)
    mch = ax.hist(mc*10**scale,bins = bins,density = density, histtype = t, cumulative = cum, log = log, label = 'MCF10A' ,linewidth = 3)
    tgh = ax.hist(tg*10**scale,bins = bins,density = density, histtype = t, cumulative = cum, log = log, label = 'MCF10A + TGF$\\beta$' ,linewidth = 3)
    ma = np.max([mdh[0].max(),mch[0].max(),tgh[0].max()])
    ax.set_prop_cycle(None)
    ax.plot(np.array([np.mean(md),np.mean(md)])*10**scale,np.array([0,ma+0.1]),linewidth = 1)
    ax.plot(np.array([np.mean(mc),np.mean(mc)])*10**scale,np.array([0,ma+0.1]),linewidth = 1)
    ax.plot(np.array([np.mean(tg),np.mean(tg)])*10**scale,np.array([0,ma+0.1]),linewidth = 1)
    ax.plot()
    plt.legend(frameon = False)
    sc_str = '10$^{-'+str(scale)+'}$'
    ax.set_xlabel('Negative Event Rate per Cell (x'+sc_str+' s$^{-1}$)')
    ax.set_ylabel('Proportion of cells')
    
    
    return fig1
        
def func(x):
    return np.mean(x)
print('Todo - add cell swarmplot?')
f = plot_MCF_summary(df,function = func)