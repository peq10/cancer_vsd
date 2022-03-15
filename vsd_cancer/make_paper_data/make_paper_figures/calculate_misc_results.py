#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 18:33:57 2021

@author: peter
"""


from pathlib import Path
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def calculate_results(save_dir,figure_dir):
    calculate_proportion_active_FOV(save_dir,figure_dir)
    
    return 0



def calculate_proportion_active_FOV(save_dir,figure_dir):
    
    df = pd.read_csv(Path(save_dir,'non_ttx_active_df_by_cell.csv'))
    df2 = pd.read_csv(Path(save_dir,'TTX_active_df_by_cell.csv'))
    
    df2 = df2[df2.stage == 'pre']
    
    df = pd.concat([df,df2])
    
    df['active'] =  (df.n_neg_events + df.n_pos_events) > 0
    df['prop_pos'] =  df.n_pos_events/(df.n_neg_events + df.n_pos_events)
    df['prop_neg'] =  df.n_neg_events/(df.n_neg_events + df.n_pos_events)
    df['day_slip'] = df.day.astype(str) + '_' + df.slip.astype(str)
    
    df['neg_event_rate'] = (df['n_neg_events'] )/(df['obs_length']*0.2)
    
    mda = df[['MCF' not in x for x in df.expt]]
    prop_active_mda = mda[['active','neg_event_rate','prop_pos','prop_neg','day_slip','expt']].groupby(['day_slip','expt']).agg(['mean'])
    
    with open(Path(figure_dir,'231_figure/proportion_active.txt'),'w') as f:
        f.write(f'{datetime.datetime.now()}\n')
        f.write(f'Mean proportion active: {100*float(prop_active_mda.active.mean()):.2f} %, ')
        f.write(f'SEM: {100*float(prop_active_mda.active.sem()):.2f} %\n')        
        f.write(f'proportion negative: {100*float(prop_active_mda.prop_neg.mean()):.2f} %\n')
        f.write(f'proportion positive: {100*float(prop_active_mda.prop_pos.mean()):.2f} %\n')
        f.write(f'Number of coverslips: {len(prop_active_mda.active)}\n')
    
    mcf = df[[x == 'MCF10A' for x in df.expt]]
    prop_active_mcf = mcf[['active','neg_event_rate','prop_pos','prop_neg','day_slip','expt']].groupby(['day_slip','expt']).agg(['mean'])
    
    tgf = df[[x == 'MCF10A_TGFB' for x in df.expt]]
    prop_active_tgf = tgf[['active','neg_event_rate','prop_pos','prop_neg','day_slip','expt']].groupby(['day_slip','expt']).agg(['mean'])
    
    with open(Path(figure_dir,'10A_figure/proportion_active.txt'),'w') as f:
        f.write(f'{datetime.datetime.now()}\n')
        f.write(f'MCF Mean proportion active: {100*float(prop_active_mcf.active.mean()):.2f} %, ')
        f.write(f'SEM: {100*float(prop_active_mcf.active.sem()):.2f} %\n')
        f.write(f'Number of MCF coverslips: {len(prop_active_mcf.active)}\n')
        
        f.write(f'TGF Mean proportion active: {100*float(prop_active_tgf.active.mean()):.2f} %, ')
        f.write(f'SEM: {100*float(prop_active_tgf.active.sem()):.2f} %\n')
        f.write(f'Number of TGF coverslips: {len(prop_active_tgf.active)}\n')
    
def calculate_number_per_fov(save_dir,figure_dir):
    df = pd.read_csv(Path(save_dir,'..','long_acqs_20210428_experiments_correct.csv'))
    
    all_densities = []
    
    for data in df.itertuples():
        if data.use != 'y':
            continue
        
        if 'MCF' in data.expt:
            continue
        
        trial_save = Path(save_dir,'ratio_stacks',data.trial_string)
        
        seg = np.load(Path(trial_save,f'{data.trial_string}_seg.npy'))
        
        seg = seg[50:-50,50:-50] # avoid edges
        n_cells = seg.max() - 1
        
        area = seg.shape[0]*seg.shape[1]*1.04**2
        
        area_mm = area * ((10**-3)**2)
        
        cells_per_mm = n_cells/area_mm
        
        all_densities.append(cells_per_mm)
    
    
    all_densities = np.array(all_densities)
    print(np.percentile(all_densities,25))
    print(np.percentile(all_densities,75))
    print(np.median(all_densities))
    
    with open(Path(figure_dir,'231_figure/cell_densities.txt'),'w') as f:
        f.write(f'{datetime.datetime.now()}\n')
        f.write(f'25 percentile per mm: {np.percentile(all_densities,25)}\n')
        f.write(f'75 percentile per mm: {np.percentile(all_densities,75)}\n')
        f.write(f'n per mm: {np.median(all_densities)}\n')

        
        
    
    
if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    calculate_results(save_dir,figure_dir)
    calculate_number_per_fov(save_dir,figure_dir)