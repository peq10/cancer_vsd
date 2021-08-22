#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:44:19 2021

@author: peter
"""

from pathlib import Path
import matplotlib.pyplot as plt

top_dir = Path('/home/peter/data/Firefly/cancer')
save_dir = Path(top_dir,'analysis','full')
figure_dir = Path('/home/peter/Dropbox/Papers/cancer/v2/')
initial_df = Path(top_dir,'analysis','long_acqs_20210428_experiments_correct.csv')

filetype = '.pdf'

#figure 1
if __name__ == '__main__':
    import method_figure
    method_figure.make_figures(initial_df,save_dir,figure_dir,filetype = filetype)
    plt.show()
    
    import patch_figure
    patch_figure.make_figures(figure_dir,filetype = filetype)
    plt.show()
    #figure 3
    #import catch22_method_figure
    #catch22_method_figure.make_figures(figure_dir,filetype = filetype)
    
    import mda_231_figure
    mda_231_figure.make_figures(initial_df,save_dir,figure_dir,filetype = filetype)
    plt.show()
    
    import ttx_figure
    ttx_figure.make_figures(initial_df,save_dir,figure_dir,filetype = filetype, redo_stats = False)
    plt.show()
    
    import MCF10A_figure
    MCF10A_figure.make_figures(initial_df,save_dir,figure_dir,filetype = filetype, redo_stats = False)
    plt.show()
    import wave_figure
    wave_figure.make_figures(top_dir,save_dir,figure_dir,filetype = filetype)
    plt.show()
else:
    
    from . import method_figure
    method_figure.make_figures(initial_df,save_dir,figure_dir,filetype = filetype)
    
    
    from . import patch_figure
    patch_figure.make_figures(figure_dir,filetype = filetype)
    
    #figure 3
    #import catch22_method_figure
    #catch22_method_figure.make_figures(figure_dir,filetype = filetype)
    
    from . import mda_231_figure
    mda_231_figure.make_figures(initial_df,save_dir,figure_dir,filetype = filetype)
    
    
    from . import ttx_figure
    ttx_figure.make_figures(initial_df,save_dir,figure_dir,filetype = filetype)
    
    
    from . import MCF10A_figure
    MCF10A_figure.make_figures(initial_df,save_dir,figure_dir,filetype = filetype)
    
    from . import wave_figure
    wave_figure.make_figures(initial_df,save_dir,figure_dir,filetype = filetype)
    plt.show()