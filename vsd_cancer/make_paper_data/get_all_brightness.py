#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 14:19:11 2021

@author: peter
"""
import numpy as np
from pathlib import Path
import pandas as pd
import tifffile


from vsd_cancer.functions import cancer_functions as canf


def get_mean_brightness(df_file, save_dir, HPC_num=None):
    df = pd.read_csv(df_file)

    for idx, data in enumerate(df.itertuples()):
        if HPC_num is not None:  # allows running in parallel on HPC
            if idx != HPC_num:
                continue
        trial_string = data.trial_string
        trial_save = Path(save_dir, "ratio_stacks", trial_string)
        seg = np.load(Path(trial_save, f"{trial_string}_seg.npy"))
        seg = seg > 0
        wh = np.where(seg)
        stack = tifffile.imread(data.tif_file, key=np.arange(8))
        mean_brightness = np.mean(stack[..., wh[0], wh[1]]) - 90 * 16
        np.save(
            Path(trial_save, f"{trial_string}_mean_brightness.npy"), mean_brightness
        )
