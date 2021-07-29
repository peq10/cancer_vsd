#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:33:36 2021

@author: peter
"""
from pathlib import Path


def make_figures(initial_df,save_dir,figure_dir,filetype = '.png'):
    figsave = Path(figure_dir,'10A_figure')
    if not figsave.is_dir():
        figsave.mkdir()
    
    
    pass



if __name__ == '__main__':
    top_dir = Path('/home/peter/data/Firefly/cancer')
    save_dir = Path(top_dir,'analysis','full')
    figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
    initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')
    make_figures(initial_df,save_dir,figure_dir)