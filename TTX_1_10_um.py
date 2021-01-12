#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:57:03 2021

@author: peter
"""
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

figsave = Path(Path.home(),'Dropbox/Papers/cancer/v1/TTX_1_10')
if not figsave.is_dir():
    figsave.mkdir(parents = True)

df = df[(df.use == 'y') & ((df.expt == 'TTX_10um')|(df.expt == 'TTX_1um'))]


trial_string = df.iloc[0].trial_string
n_thresh = len(np.load(Path(Path(save_dir,'ratio_stacks',trial_string),f'{trial_string}_event_properties.npy'),allow_pickle = True).item()['events'])


currents  = [[[],[],[],[]] for x in range(n_thresh)]
lengths  = [[[],[],[],[]] for x in range(n_thresh)]

#here we do not differentiate buy slip, just using all cells
for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string

    trial_save = Path(save_dir,'ratio_stacks',trial_string)
    
    results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()

    cell_ids = np.arange(results['events'][0]['tc_filt'].shape[0])
    cell_ids = [x for x in cell_ids if x not in results['excluded_circle']]

    for idx,thresh_level_dict in enumerate(results['events']):
    
        
        
        event_props = results['events'][idx]['event_props']
        
        
        if data.stage == 'pre' and '10um' in data.expt:
            idx2 = 0
        elif data.stage == 'post' and '10um' in data.expt:
            idx2 = 1
        elif data.stage == 'post' and '1um' in data.expt:
            print(data.trial_string)
            idx2 = 2
        elif data.stage == 'pre' and '1um' in data.expt:
            idx2 = 3
        elif data.stage == 'washout':
            continue
        else:
            raise ValueError('oops')
        
        observations = [results['observation_length'][idx][x] for x in cell_ids]
        sum_current = [np.sum(np.abs(event_props[x][:,-1])) if x in event_props.keys() else 0 for x in cell_ids]

        
        currents[idx][idx2].extend(sum_current)
        lengths[idx][idx2].extend(observations)
        
pre_10_current = [np.sort(np.array(p[0])) for p in currents if len(p[0]) != 0]
pre_10_length = [np.array(x[0]) for x in lengths]
post_10_current = [np.array(p[1]) for p in currents if len(p[1]) != 0]
post_10_length = [np.array(x[1]) for x in lengths]
post_1_current = [np.array(p[2]) for p in currents if len(p[2]) != 0]
post_1_length = [np.array(x[2]) for x in lengths]
pre_1_current = [np.array(p[3]) for p in currents if len(p[3]) != 0]
pre_1_length = [np.array(x[3]) for x in lengths]

#adjust for observation length
pre_10_adj = [pre_10_current[i]/pre_10_length[i] for i in range(len(pre_10_current))]
post_10_adj = [post_10_current[i]/post_10_length[i] for i in range(len(post_10_current))]
post_1_adj = [post_1_current[i]/post_1_length[i] for i in range(len(post_1_current))]
pre_1_adj = [pre_1_current[i]/pre_1_length[i] for i in range(len(pre_1_current))]

#threshold level
idx = 2

#normalise the integrals to 1 max
ma_10 = np.max([pre_10_adj[idx].max(),post_10_adj[idx].max()])
ma_1 = np.max([post_1_adj[idx].max(),pre_1_adj[idx].max()])
pre_10_adj[idx] /= ma_10
post_10_adj[idx] /= ma_10
post_1_adj[idx] /= ma_1
pre_1_adj[idx] /= ma_1

#first plot the median, IQR of non-zero and proportion of zero current cells
nonzer_curr = [pre_10_adj[idx][pre_10_adj[idx]!=0],post_10_adj[idx][post_10_adj[idx]!=0],pre_1_adj[idx][pre_1_adj[idx]!=0],post_1_adj[idx][post_1_adj[idx]!=0]]
all_curr = [pre_10_adj[idx],post_10_adj[idx],pre_1_adj[idx],post_1_adj[idx]]


n_bins = 'blocks'


density = True
cumulative =  True
log = True

fig1,ax1 = plt.subplots()

linewidth = 3
pres_10_hist = av.hist(pre_10_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative,color = 'r',log = log,linewidth = linewidth,label = 'Pre 10 TTX')

post_10_hist = av.hist(post_10_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative, color = 'b',log = log,linewidth = linewidth,label = 'TTX 10 $\mathrm{\mu}$M')


plt.legend(loc = (0.5,0.1),frameon = False,fontsize = 14)
ax1.set_xlabel('Integrated activity per cell (a.u.)')
ax1.set_ylabel('(Log) Cell fraction')
pf.set_all_fontsize(ax1, 14)
pf.set_thickaxes(ax1, 3)
ax1.tick_params(which = 'minor',width = 3,length = 3)
#ax1.set_yticks([0.9,1])
#ax1.set_ylim([0.9,1])
ax1.minorticks_off()
fig1.savefig(Path(figsave,'cumulative_log_histogram_10uM.png'),bbox_inches = 'tight',dpi = 300)

fig1b,ax1b = plt.subplots()

linewidth = 3
post_1_hist = av.hist(post_1_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative,color = 'g',log = log,linewidth = linewidth,label = 'TTX 1 $\mathrm{\mu}$M')
pres_1_hist = av.hist(pre_1_adj[idx], histtype='step', bins=100, density=density ,
                     cumulative=cumulative,color = 'k',log = log,linewidth = linewidth,label = 'Pre 1 TTX')

plt.legend(loc = (0.2,0.1),frameon = False,fontsize = 14)
ax1.set_xlabel('Integrated activity per cell (a.u.)')
ax1.set_ylabel('(Log) Cell fraction')
pf.set_all_fontsize(ax1, 14)
pf.set_thickaxes(ax1, 3)
ax1.minorticks_off()
ax1.tick_params(which = 'minor',width = 3,length = 3)
fig1b.savefig(Path(figsave,'cumulative_log_histogram_1uM.png'),bbox_inches = 'tight',dpi = 300)


#now also just plot the means and standard deviations
means = np.array([np.mean(x) for x in all_curr])
#bootstrap a confidence interval on the mean

boot_samples = np.array([ass.bootstrap(x,bootnum = 1000,bootfunc = np.mean) for x in all_curr])
boot_range = np.array([np.percentile(boot_samples,5,axis = -1),np.percentile(boot_samples,95,axis = -1)])

ma_10 = means[:2].max()
ma_1 = means[2:].max()
means[:2] /= ma_10
means[2:] /= ma_1
boot_range[:,:2] /= ma_10
boot_range[:,2:] /= ma_1

boot_err = np.array([means - boot_range[0,:],boot_range[1,:] - means])

fig3,ax3 = plt.subplots()
ax3.errorbar(np.arange(2),means[:2],yerr = boot_err[:,:2],color = 'k',marker = 's',mfc = 'k',linewidth = 2,capthick = 2,elinewidth = 2,capsize = 5)
ax3.errorbar(np.arange(2)+2,means[2:],yerr = boot_err[:,2:],color = 'k',marker = 's',mfc = 'k',linewidth = 2,capthick = 2,elinewidth = 2,capsize = 5)

#ax3.plot(np.arange(3),boot_range.T)
#ax3.set_ylim([0,1.6])
ax3.set_xticks(np.arange(4))
ax3.set_xticklabels(['Pre TTX','TTX 10 $\mathrm{\mu}$m','pre TTX 1 $\mathrm{\mu}$m','TTX 1 $\mathrm{\mu}$m'])
ax3.set_ylabel('Mean activity per cell\n(normalised)')
pf.set_all_fontsize(ax3, 16)
pf.set_thickaxes(ax3, 3)
fig3.savefig(Path(figsave,'all_mean_stderr.png'),bbox_inches = 'tight',dpi = 300)


num_zers = np.array([np.sum(pre_10_adj[idx]==0),np.sum(post_10_adj[idx]==0),np.sum(pre_1_adj[idx]==0),np.sum(post_1_adj[idx]==0)])
num_cells_tot = np.array([len(pre_10_adj[idx]),len(post_10_adj[idx]),len(pre_1_adj[idx]),len(post_1_adj[idx])])

proportion_zero = num_zers/num_cells_tot

fig2,ax2 = plt.subplots()

scale = 0.3
ax2.violinplot(nonzer_curr)
for i in range(4):
    ax2.plot(np.linspace(-scale,scale,len(nonzer_curr[i]))+i+1,np.sort(nonzer_curr[i]),'.',color = 'k',alpha = 0.5)

ax2.set_ylabel('Non-zero event size')
ax2b = ax2.twinx()
ax2b.plot(np.arange(2)+1,proportion_zero[:2],'-r',marker = '.',markersize = 15)
ax2b.plot(np.arange(2)+3,proportion_zero[2:],'-r',marker = '.',markersize = 15)
ax2b.tick_params(axis='y', labelcolor='r')
ax2b.set_ylabel('Proportion of inactive cells', color='r')  #
ax2.set_xticks(np.arange(4)+1)
ax2.set_xticklabels(['Pre TTX','TTX 10 $\mathrm{\mu}$m','Pre TTX 1 um','TTX 1 um'])
pf.set_all_fontsize(ax2, 16)
pf.set_all_fontsize(ax2b, 16)
fig2.savefig(Path(figsave,'nonzero_violin.png'),bbox_inches = 'tight',dpi = 300)


#finally do tests on full histos
test1 = stats.anderson_ksamp([all_curr[0],all_curr[1]])
test2 = stats.anderson_ksamp([all_curr[2],all_curr[3]])
test3 = stats.anderson_ksamp([all_curr[0],all_curr[2]])

print(f'Pre-10 p = {test1.significance_level}')
print(f'Pre-1 p = {test2.significance_level}')
print(f'Pre-pre p = {test3.significance_level}')

print(f'cells 10 uM: {[len(x) for x in all_curr[:2]]}')

print(f'cells 1 uM: {[len(x) for x in all_curr[-1:1:-1]]}')
