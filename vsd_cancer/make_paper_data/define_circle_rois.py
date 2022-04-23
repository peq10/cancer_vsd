#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:10:38 2021

@author: peter
"""
# look at excluding ROIs outside the central column

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from vsd_cancer.functions import cancer_functions as canf


def define_circle_rois(
    top_dir,
    initial_df,
    save_dir,
    radius=220,
    center=(246, 256),
    HPC_num=None,
    redo=False,
):
    # TODO only save once to avoid race condition
    if HPC_num is not None:  # only want it saving once
        if (
            Path(
                save_dir,
                f"{initial_df.stem}_intermediate_files",
                f"{initial_df.stem}_roi_df.csv",
            ).is_file()
            and not redo
        ):
            return 0

    df = pd.read_csv(initial_df)

    rois = []
    for idx, data in enumerate(df.itertuples()):
        fname = Path(data.tif_file)
        meta = canf.load_tif_metadata(fname)
        roi = meta["FrameKey-0-0-0"]["ROI"]
        rois.append(np.array(roi.split("-")).astype(int))

    rois = np.array(rois)

    # now specify x,y position
    circle_roi_centers = np.array(center) - rois[:, :2]

    roi_df = df.loc[:, ["trial_string"]]
    roi_df["capture_roi_x"] = rois[:, 0]
    roi_df["capture_roi_y"] = rois[:, 1]
    roi_df["capture_roi_width"] = rois[:, 2]
    roi_df["capture_roi_height"] = rois[:, 3]
    roi_df["circle_roi_center_x"] = circle_roi_centers[:, 0]
    roi_df["circle_roi_center_y"] = circle_roi_centers[:, 1]
    roi_df["circle_roi_radius"] = radius

    roi_df.to_csv(
        Path(
            save_dir,
            f"{initial_df.stem}_intermediate_files",
            f"{initial_df.stem}_roi_df.csv",
        )
    )
