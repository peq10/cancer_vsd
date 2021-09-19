#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 10:35:44 2021

@author: peter
"""
import pandas as pd 
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from vsd_cancer.functions import cancer_functions as canf

top_dir = Path('/home/peter/data/Firefly/cancer')
df_str = ''
save_dir = Path(top_dir,'analysis','full')


df1 = pd.read_csv(Path('/home/peter/data/Firefly/cancer/analysis/full/good_detections.csv'))


df2 = pd.read_csv(Path('/home/peter/data/Firefly/cancer/analysis/full/yilin_data/yilin_new/all_cell_labels_yilin.csv'))

#df1 = df1[df1.correct == True]
#df2 = df2[df2.video_new == 'y']


df1['cid'] = df1.trial_string + '_' + df1.cell_id.astype(int).astype(str)
df2['cid'] = df2.trial_string + '_' + df2.id.astype(int).astype(str)



mine = [x for x in df1.cid]
yilin = [x for x in df2.cid]


all_cells = list(np.unique(mine + yilin))

only_me = [x for x in mine if x not in yilin]

only_yilin = [x for x in yilin if x not in mine]

'''

trials = np.unique([x[:x.rfind('_')] for x in all_cells])
for t in trials:
    
    trial_save = Path(save_dir,'ratio_stacks',t)
    my_seg = np.load(Path(trial_save,f'{t}_seg.npy'))
    
    try:
        yilin_seg = np.load(Path(save_dir,'yilin_data/send_to_peter2/send_to_peter',f'{t}_seg.npy'))
    except:
        print('not found')
        continue
    print(np.all(my_seg == yilin_seg))
    plt.imshow(np.logical_xor(my_seg > 0,yilin_seg >0))
    plt.show()
            
    
    
    
    continue
    
    my_cells = [int(x[x.rfind('_')+1:]) for x in mine if t in x]
    yilins_cells = [int(x[x.rfind('_')+1:]) for x in yilin if t in x]

    if my_cells == yilins_cells:
        continue
    
    
    if np.all([x in my_cells for x in yilins_cells]):
        continue
    
    plt.plot(my_cells)
    plt.plot(yilins_cells)
    print('Different')
    #plt.show()
    trial_save = Path(save_dir,'ratio_stacks',t)
    seg = np.load(Path(trial_save,f'{t}_seg.npy'))
    if np.max(yilins_cells) > seg.max():
        print('Over')
    
    if False:
            
        trial_save = Path(save_dir,'ratio_stacks',t)
        seg = np.load(Path(trial_save,f'{t}_seg.npy'))
        masks = canf.lab2masks(seg)
        
        masks_me = masks[my_cells]
        masks_yilin = masks[yilins_cells]
        
        plt.imshow(np.sum(masks_me,0) + np.sum(masks_yilin,0)*2)
        plt.show()







#Plot the segs of the only yilin ones

trials = np.unique([x[:x.rfind('_')] for x in only_yilin])


for t in trials:
    cells = [int(x[x.rfind('_')+1:]) for x in only_yilin if t in x]
    
    trial_save = Path(save_dir,'ratio_stacks',t)
    seg = np.load(Path(trial_save,f'{t}_seg.npy'))
    masks = canf.lab2masks(seg)
    
    masks = masks[cells]
    
    plt.imshow(np.sum(masks,0))
    plt.show()
'''

