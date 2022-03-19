#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:09:41 2021

@author: peter
"""
# a script to work out which ROIs are included in the circle roi

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from vsd_cancer.functions import cancer_functions as canf


def apply_circle_exclusion(top_dir, save_dir, initial_df):

    df = pd.read_csv(Path(save_dir, "roi_df.csv"))
    for idx, data in enumerate(df.itertuples()):
        trial_string = data.trial_string
        print(trial_string)

        trial_save = Path(save_dir, "ratio_stacks", trial_string)

        seg = np.load(Path(trial_save, f"{trial_string}_seg.npy"))

        # define an exclusion zone
        y, x = np.indices(seg.shape)

        y -= data.circle_roi_center_y
        x -= data.circle_roi_center_x

        r = np.sqrt(x**2 + y**2)
        exc = r > data.circle_roi_radius

        masks = canf.lab2masks(seg)
        mask_sz = np.sum(masks, axis=(-2, -1))

        intersect_idx = np.argwhere(
            np.sum(np.logical_and(exc[None, ...], masks), axis=(-2, -1)) > 0.1 * mask_sz
        )

        np.save(
            Path(trial_save, f"{trial_string}_circle_excluded_rois.npy"),
            intersect_idx.ravel(),
        )

        plt.imshow(seg + exc * seg.max())
        plt.show()


if __name__ == "__main__":
    top_dir = Path("/home/peter/data/Firefly/cancer")
    save_dir = Path(top_dir, "analysis", "full")
    initial_df = Path(top_dir, "analysis", "long_acqs_20210428_experiments_correct.csv")
    apply_circle_exclusion(top_dir, save_dir, initial_df)
