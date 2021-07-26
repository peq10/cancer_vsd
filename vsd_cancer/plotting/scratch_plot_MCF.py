#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:16:49 2021

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import scipy.stats as stats

import astropy.visualization as av

from pathlib import Path

import f.plotting_functions as pf

import seaborn as sns

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)

figsave = Path(Path.home(),'Dropbox/papers/cancer/v1/MCF_V_TGB/')
if not figsave.is_dir():
    figsave.mkdir(parents = True)

df = df[(df.use == 'y') & ((df.expt == 'MCF10A')|(df.expt == 'MCF10A_TGFB')|(df.expt == 'standard'))]


slip_currents = []

slip_num_events = []

n_cells = []

observation_length = []

slips = []

days = []

expt = []

area = []
day_slip = []
#here we do not differentiate buy slip, just using all cells
for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    #print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)

    if data.expt != 'standard':
        results =  np.load(Path(trial_save,f'{trial_string}_event_properties_including_user_input.npy'),allow_pickle = True).item()
    else:
        results = np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
        results['events'] = results['events'][1]
    
    active_cells = [x for x in results['events'].keys() if type(x) != str]
    
    slip_num_events.append(np.sum([results['events'][x].shape[-1] for x in active_cells]))
    
    slip_currents.append(np.sum([np.sum(np.abs(results['events']['event_props'][x][:,-1])) for x in active_cells]))
    
    n_cells.append(np.nonzero(results['observation_length'])[0].shape[0])
    
    observation_length.append(np.sum(results['observation_length']))
    
    slips.append(data.slip)
    
    days.append(data.date)
    
    expt.append(data.expt)
    
    area.append(data.area)
    
    
    day_slip.append(f'{data.date}{data.slip}')
    
res = pd.DataFrame({'day':days,'slips':slips,'day_slip':day_slip,'currents':slip_currents,'n_events':slip_num_events,'obs_len':observation_length,'n_cells':n_cells,'expt':expt})
    
#calculate mean current per cell time point
res['norm_curr'] = res['currents']/res['obs_len']


means = res.groupby(['expt','day'], as_index=False).agg({'norm_curr': "mean"})
#make a superplot of 
sns.swarmplot(x = 'expt',y = 'norm_curr',data = res,hue = 'day')
ax = sns.swarmplot(x = 'expt',y = 'norm_curr',data = means,hue = 'day',size=10, edgecolor="k", linewidth=2)
ax.legend_.remove()

mean_mcf = means['norm_curr'].values[:4]
mean_tgb = means['norm_curr'].values[4:]

stat_res = stats.ttest_ind(mean_mcf,mean_tgb)


x1, x2 = 0, 1
y, h, col = res['norm_curr'].max()*1.1, res['norm_curr'].max()/10, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col);
plt.text((x1+x2)*.5, y+h*1.4, f"p = {stat_res.pvalue:.3f}", ha='center', va='bottom', color=col)


slips = res.groupby(['expt','day_slip','day'], as_index=False).agg({'norm_curr': "mean"})

plt.show()
ax2 = sns.swarmplot(x = 'expt',y = 'norm_curr',data = slips,hue =None,color = 'k',size = 8,order = ['standard','MCF10A','MCF10A_TGFB'])
ax2.set_xticklabels(['MDA MB\n231', 'MCF10A', 'MCF10A+\n'+r'TGF$\mathrm{\beta}$'],rotation = 45)
ax2.set_xlabel('Cell Type')
ax2.set_ylabel('Mean Vm Activity (a.u.)')

pf.set_all_fontsize(ax2, 16)

pf.set_thickaxes(ax2, 3)
ax2.set_xticklabels(['MDA MB\n231', 'MCF10A', 'MCF10A+\n'+r'TGF$\mathrm{\beta}$'],rotation = 45,fontsize = 16)
pf.make_square_plot(ax2)
plt.savefig(Path(Path.home(),'Dropbox/Papers/cancer/Wellcome/standard_vs_mda.png'),bbox_inches = 'tight',dpi = 300,transparent = True)

'''
slip_mcf = slips['norm_curr'].values[slips['expt'] == 'MCF10A']
slip_tgb = slips['norm_curr'].values[slips['expt'] == 'MCF10A_TGFB']
slip_standard = slips['norm_curr'].values[slips['expt'] == 'standard']

stats2 = stats.mannwhitneyu(slip_tgb,slip_mcf)

x1, x2 = 0, 1
y, h, col = slip_tgb.max()*1.1, res['norm_curr'].max()/10, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col);
plt.text((x1+x2)*.5, y+h*1.4, f"p = {stats2.pvalue:.3f}", ha='center', va='bottom', color=col)
'''