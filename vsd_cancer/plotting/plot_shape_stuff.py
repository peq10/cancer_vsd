#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 20:14:08 2021

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

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
HPC_num = None


save_dir = Path(top_dir,'analysis','full')
viewing_dir = Path(top_dir,'analysis','full','tif_viewing','videos')
initial_df = Path(top_dir,'analysis',f'long_acqs_20210428_experiments_correct{df_str}.csv')

df = pd.read_csv(initial_df)


df = df[(df.use == 'y') & (df.expt == 'standard')]


circularities = []
currents = []

active_circ = []
inactive_circ = []

thresh_idx = 1
#here we do not differentiate buy slip, just using all cells
for idx,data in enumerate(df.itertuples()):
    trial_string = data.trial_string
    #print(trial_string)
    trial_save = Path(save_dir,'ratio_stacks',trial_string)

    
    results =  np.load(Path(trial_save,f'{trial_string}_event_properties.npy'),allow_pickle = True).item()
    events = results['events'][thresh_idx]
    
    active_cells = [x for x in events.keys() if type(x) != str]
    
    n_cells = results['n_cells']
    inactive_cells = [x for x in range(n_cells) if x not in active_cells]
    
    curr = [np.sum(np.abs(events['event_props'][x][:,-1])) if x in active_cells else 0 for x in range(n_cells)]
    
    currents += curr
    morph = pd.read_csv(Path(trial_save,f'{trial_string}_cell_morphologies.csv'))
    
    for x in range(n_cells):
        circ = morph[morph.cell == x]['circularity'].values[0]
        if circ >1:
            circ = 1
        
        circularities.append(circ)
        
        if x in active_cells:
            active_circ.append(circ)
        else:
            inactive_circ.append(circ)
            
currents = np.array(currents)
circs = np.array(circularities)

active_circ = np.array(active_circ)
inactive_circ = np.array(inactive_circ)
        
wh = np.where(currents >-1)[0]

tst1 =currents[wh]
tst2 = circs[wh]

plt.plot(tst2,tst1,'.')
plt.show()


m1 = currents
m2 = circs

xmin = m1.min()

xmax = m1.max()

ymin = m2.min()

ymax = m2.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

positions = np.vstack([X.ravel(), Y.ravel()])

values = np.vstack([m1, m2])

kernel = stats.gaussian_kde(values)

Z = np.reshape(kernel(positions).T, X.shape)

fig, ax = plt.subplots()

ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,

          extent=[xmin, xmax, ymin, ymax])

ax.plot(m1, m2, 'k.', markersize=2)

ax.set_xlim([xmin, xmax])

ax.set_ylim([ymin, ymax])

plt.show()
plt.violinplot([inactive_circ,active_circ])


fig,ax = plt.subplots()
ax.plot(circs,currents,'o',markersize = 7,mfc = 'k',mew = 0,alpha = 0.7)
ax.set_xlabel('Circularity')
ax.set_ylabel('Single Cell Vm activity (a.u.)')

pf.make_square_plot(ax)
pf.set_all_fontsize(ax, 16)
pf.set_thickaxes(ax, 3)

fig.savefig(Path(Path.home(),'Dropbox/Papers/cancer/Wellcome/circularity.png'),bbox_inches = 'tight',dpi = 300)