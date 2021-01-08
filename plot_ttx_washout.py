#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:21:19 2021

@author: peter
"""
#a script to plot the TTX 10 um trials

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats

import astropy.visualization as av
import astropy.stats as ass

from pathlib import Path

import f.plotting_functions as pf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20201230_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/TTX_washout')
if not figsave.is_dir():
    figsave.mkdir(parents = True)

df = df[(df.use == 'y') & (df.expt == 'TTX_10um_washout')]


trial_string = df.iloc[0].trial_string
n_thresh = len(np.load(Path(Path(save_dir,'ratio_stacks',trial_string),f'{trial_string}_event_properties.npy'),allow_pickle = True).item()['events'])

currents  = [[[],[],[]] for x in range(n_thresh)]
lengths  = [[[],[],[]] for x in range(n_thresh)]

#here we do not differentiate buy slip, just using all cells
for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

    cell_ids = np.arange(results['events'][0]['tc_filt'].shape[0])
    cell_ids = [x for x in cell_ids if x not in results['excluded_circle']]

    for idx,thresh_level_dict in enumerate(results['events']):
    
        
        
        event_props = results['events'][idx]['event_props']
        
        
        if data.stage == 'pre':
            idx2 = 0
        elif data.stage == 'post':
            idx2 = 1
        elif data.stage == 'washout':
            idx2 = 2
        else:
            raise ValueError('oops')
        
        observations = [results['observation_length'][idx][x] for x in cell_ids]
        sum_current = [np.sum(np.abs(event_props[x][:,-1])) if x in event_props.keys() else 0 for x in cell_ids]

        
        currents[idx][idx2].extend(sum_current)
        lengths[idx][idx2].extend(observations)
            

    
pre_current = [np.array(p[0]) for p in currents if len(p[0]) != 0]
pre_length = [np.array(x[0]) for x in lengths]
post_current = [np.array(p[1]) for p in currents if len(p[1]) != 0]
post_length = [np.array(x[1]) for x in lengths]
wash_current = [np.array(p[2]) for p in currents if len(p[2]) != 0]
wash_length = [np.array(x[2]) for x in lengths]

#adjust for observation length
pre_adj = [pre_current[i]/pre_length[i] for i in range(len(pre_current))]
post_adj = [post_current[i]/post_length[i] for i in range(len(post_current))]
wash_adj = [wash_current[i]/wash_length[i] for i in range(len(wash_current))]

#threshold level
idx = 3

#normalise the integrals to 1 max
ma = np.max([pre_adj[idx].max(),post_adj[idx].max(),wash_adj[idx].max()])
pre_adj[idx] /= ma
post_adj[idx] /= ma
wash_adj[idx] /= ma

#first plot the median, IQR of non-zero and proportion of zero current cells
nonzer_curr = [pre_adj[idx][pre_adj[idx]!=0],post_adj[idx][post_adj[idx]!=0],wash_adj[idx][wash_adj[idx]!=0]]
all_curr = [pre_adj[idx],post_adj[idx],wash_adj[idx]]

medians = np.array([np.median(x) for x in nonzer_curr])
IQRs = np.array([[np.percentile(x,25),np.percentile(x,75)] for x in nonzer_curr])

num_zers = np.array([np.sum(pre_adj[idx]==0),np.sum(post_adj[idx]==0),np.sum(wash_adj[idx]==0)])
num_cells_tot = np.array([len(pre_adj[idx]),len(post_adj[idx]),len(wash_adj[idx])])

proportion_zero = num_zers/num_cells_tot



fig2,ax2 = plt.subplots()
#ax2.errorbar(np.arange(3),medians,yerr = IQRs.T,color = 'k',marker = 's',mfc = 'k')
scale = 0.3
ax2.violinplot(nonzer_curr)
for i in range(3):
    ax2.plot(np.linspace(-scale,scale,len(nonzer_curr[i]))+i+1,np.sort(nonzer_curr[i]),'.',color = 'k',alpha = 0.5)

ax2.set_ylabel('Non-zero event size')
ax2b = ax2.twinx()
ax2b.plot(np.arange(3)+1,proportion_zero,'-r',marker = '.',markersize = 15)
ax2b.tick_params(axis='y', labelcolor='r')
ax2b.set_ylabel('Proportion of inactive cells', color='r')  #
ax2.set_xticks(np.arange(3)+1)
ax2.set_xticklabels(['Pre TTX','TTX 10 $\mathrm{\mu}$m','Washout'])
pf.set_all_fontsize(ax2, 16)
pf.set_all_fontsize(ax2b, 16)
fig2.savefig(Path(figsave,'nonzero_violin.png'),bbox_inches = 'tight',dpi = 300)


#now also just plot the means and standard deviations
means = np.array([np.mean(x) for x in all_curr])
#bootstrap a confidence interval on the mean

boot_samples = np.array([ass.bootstrap(x,bootnum = 1000,bootfunc = np.mean) for x in all_curr])
boot_range = np.array([np.percentile(boot_samples,25,axis = -1),np.percentile(boot_samples,75,axis = -1)])



std_errs = np.array([np.std(x) for x in all_curr])

#normalise to 1
ma = means.max()
means /= ma
std_errs /= ma
boot_range /= ma

boot_err = np.array([means - boot_range[0,:],boot_range[1,:] - means])

fig3,ax3 = plt.subplots()
ax3.errorbar(np.arange(3),means,yerr = boot_err,color = 'k',marker = 's',mfc = 'k',linewidth = 2,capthick = 2,elinewidth = 2,capsize = 5)
#ax3.plot(np.arange(3),boot_range.T)
#ax3.set_ylim([0,1.6])
ax3.set_xticks(np.arange(3))
ax3.set_xticklabels(['Pre TTX','TTX 10 $\mathrm{\mu}$m','Washout'])
ax3.set_ylabel('Mean activity per cell\n(normalised)')
pf.set_all_fontsize(ax3, 16)
pf.set_thickaxes(ax3, 3)
fig3.savefig(Path(figsave,'all_mean_stderr.png'),bbox_inches = 'tight',dpi = 300)


n_bins = 'blocks'


density = True
cumulative =  True
log = True

fig1,ax1 = plt.subplots()

linewidth = 3
pres_hist = av.hist(pre_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative,color = 'r',log = log,linewidth = linewidth,label = 'Pre-TTX')
#pres_hist = av.hist(pre_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False,color = 'r')
post_hist = av.hist(post_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative, color = 'b',log = log,linewidth = linewidth,label = 'TTX 10 $\mathrm{\mu}$M')
#post_hist = av.hist(post_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False, color = 'b')
wash_hist = av.hist(wash_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative,color = 'g',log = log,linewidth = linewidth,label = 'Washout')
#wash_hist = av.hist(wash_current[idx][:, -1], histtype='step', bins=n_bins, density=density ,
#                     cumulative=False,color = 'g')
plt.legend(loc = (0.5,0.1),frameon = False,fontsize = 14)
ax1.set_xlabel('Integrated activity per cell (a.u.)')
ax1.set_ylabel('(Log) Cell fraction')
pf.set_all_fontsize(ax1, 14)
pf.set_thickaxes(ax1, 3)
ax1.tick_params(which = 'minor',width = 3,length = 3)
#ax1.set_yticks([0.9,1])
#ax1.set_ylim([0.9,1])
ax1.minorticks_off()
fig1.savefig(Path(figsave,'cumulative_log_histogram.png'),bbox_inches = 'tight',dpi = 300)


#finally do tests on full histos
test1 = stats.anderson_ksamp([all_curr[0],all_curr[1]])
test2 = stats.anderson_ksamp([all_curr[0],all_curr[2]])
test3 = stats.anderson_ksamp([all_curr[1],all_curr[2]])

print(f'Pre-post p = {test1.significance_level}')
print(f'Pre-wash p = {test2.significance_level}')
print(f'Post-wash p = {test3.significance_level}')


'''

#bins = np.histogram_bin_edges(pre_current[idx][:, -1], bins=n_bins)
pres_hist = av.hist(pre_current[idx][:, -1], histtype='step', bins=n_bins, density=True,
                     weights=np.ones(pre_current[idx].shape[0])/np.sum(pre_length[idx]), cumulative=True,color = 'r')
post_hist = av.hist(post_current[idx][:, -1], histtype='step', bins=n_bins, density=True,
                     weights=np.ones(post_current[idx].shape[0])/np.sum(post_length[idx]), cumulative=True, color = 'b')
wash_hist = av.hist(wash_current[idx][:, -1], histtype='step', bins=n_bins, density=True,
                     weights=np.ones(wash_current[idx].shape[0])/np.sum(wash_length[idx]), cumulative=True,color = 'g')




phist = np.histogram(np.abs(pre_current[idx][:, -1]), bins=n_bins,
                     weights=np.ones(pre_current[idx].shape[0])/np.sum(pre_length[idx]))
pohist = np.histogram(np.abs(post_current[idx][:,-1]),bins = n_bins,weights = np.ones(post_current[idx].shape[0])/np.sum(post_length[idx]))
'''