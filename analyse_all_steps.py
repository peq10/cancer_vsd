#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 18:25:53 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import cancer_functions as canf
import f.general_functions as gf
import f.plotting_functions as pf
import scipy.stats

df = pd.read_csv('/home/peter/data/Firefly/cancer/analysis/old_steps.csv')

df['old'] = True

df2 = pd.read_csv('/home/peter/data/Firefly/cancer/analysis/steps_20201230_sorted.csv')
df2 = df2[df2.run == 0]

df2['old'] = False

df = df.append(df2)

mean_fs = []
mean_vs = []
mean_is = []
mean_rs = []
fits = []
sens = []

example = 'cancer_20201113_slip1_cell1_steps_steps_with_emission_ratio_steps_green_0.125_blue_0.206_1'



for idx,data in enumerate(df.itertuples()):
    
    if data.old:
        trial_string = data.trial_string
    else:
        s = data.tif_file
        trial_string = '_'.join(Path(s).parts[Path(s).parts.index('cancer'):-1])
    trial_save = Path('/home/peter/data/Firefly/cancer/analysis/full','steps_analysis/data',trial_string)
    
    df_t = np.load(Path(trial_save,f'{trial_string}_df_tc.npy'))
    vm = np.load(Path(trial_save,f'{trial_string}_vm.npy'))
    im = np.load(Path(trial_save,f'{trial_string}_im.npy'))
    
    if data.old:
        stim_locs = np.array([42,86])
    else:
        stim_locs = np.array([25,49])
        
    mean_f = np.mean(df_t[...,stim_locs[0]:stim_locs[1]],-1)
    mean_fs.append(mean_f)
    
    dr_t = (df_t[:,0,:]+1)/(df_t[:,1,:]+1)
    
    mean_r = np.mean(dr_t[...,stim_locs[0]:stim_locs[1]],-1)
    mean_rs.append(mean_r)
    
    #plt.plot(dr_t.T)
    #plt.show()
    #print(trial_string)
    
    v_locs = np.round((stim_locs/df_t.shape[-1])*vm.shape[-1]).astype(int)
    
    mean_v = np.mean(vm[:,v_locs[0]:v_locs[1]],-1)
    mean_vs.append(mean_v)
    
    mean_i = np.mean(im[:,v_locs[0]:v_locs[1]],-1)
    mean_is.append(mean_i)
    
    fit_blue = scipy.stats.linregress(mean_v,mean_f[:,0])
    fit_green = scipy.stats.linregress(mean_v,mean_f[:,1])
    fit_rat = scipy.stats.linregress(mean_v,mean_r)
    
    fits.append([fit_blue,fit_green,fit_rat])

    sens.append([fit_blue.slope,fit_green.slope,fit_rat.slope])
    
    if trial_string == example:
        ex_tc = df_t
        ex_vm = vm
        ex_im = im
        ii = idx


mean_fs = np.array(mean_fs)
mean_vs = np.array(mean_vs)
mean_is = np.array(mean_is)
mean_rs = np.array(mean_rs)

sens = np.array(sens)

sens = sens*100**2 






#plot an example cell
fig = plt.figure(constrained_layout = True)
gs  = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0,0])



ax2 = fig.add_subplot(gs[0,1])



ax3 = fig.add_subplot(gs[1,0])
ax3.plot(mean_vs[ii,:],(fits[ii][-1].slope*mean_vs[ii,:] + fits[ii][-1].intercept - 1)*100,'k',linewidth = 3)
ax3.plot(mean_vs[ii,:],(mean_rs[ii,:]-1)*100,'.r',markersize = 12)
ax3.set_xlabel('Membrane Voltage (mV)')
ax3.set_ylabel(r'$\Delta R/R_0$ (%)')
pf.set_all_fontsize(ax3, 12)
pf.set_thickaxes(ax3, 3)




ax4 = fig.add_subplot(gs[1,1])
scale = 0.02
ax4.violinplot(sens[:,-1])
ax4.plot(np.random.normal(loc = 1,scale = scale,size = sens.shape[0]),sens[:,-1],'.k',markersize = 12)
ax4.xaxis.set_visible(False)
ax4.set_yticks(np.arange(1,6))
ax4.set_ylabel('Ratiometric sensitivity\n(% per 100 mV)')
pf.set_thickaxes(ax4, 3,remove = ['top','right','bottom'])
pf.set_all_fontsize(ax4, 12)