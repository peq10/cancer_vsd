#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:07:19 2021

@author: peter
"""
import numpy as np
from pathlib import Path
import pandas as pd

import scipy.ndimage as ndimage

import os
import tifffile
import time
import f.general_functions as gf


def make_roi_overlay(events_dict, seg, sz):
    overlay = np.zeros(sz, dtype=int)
    for idx in events_dict.keys():
        if type(idx) == str:
            continue
        for idx2 in range(events_dict[idx].shape[-1]):
            ids = events_dict[idx][:, idx2]
            mask = (seg == idx + 1).astype(int)
            outline = np.logical_xor(
                mask, ndimage.binary_dilation(mask, iterations=4)
            ).astype(int)
            overlay[ids[0] : ids[1], ...] += outline

    overlay = overlay > 0

    return overlay


redo_trials = []
redo_trials2 = [
    "cancer_20210313_slip7_area1_long_acq_MCF10A_TGFBETA_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210313_slip5_area3_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210313_slip4_area2_long_acq_corr_MCF10A_37deg_long_acq_blue_0.039_green_0.04734_1",
    "cancer_20210313_slip1_area3_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.0506_green_0.097_1",
    "cancer_20210122_slip5_area3_long_acq_MCF10A_tgfbeta_long_acq_blue_0.0378_green_0.0791_illum_feedback_on_1",
    "cancer_20210122_slip5_area2_long_acq_MCF10A_tgfbeta_long_acq_blue_0.0378_green_0.0791_illum_feedback_on_1",
    "cancer_20210122_slip4_area2_long_acq_MCF10A_tgfbeta_long_acq_blue_0.0378_green_0.0791_illum_feedback_on_2",
    "cancer_20210122_slip4_area1_long_acq_MCF10A_tgfbeta_long_acq_blue_0.0378_green_0.0791_illum_feedback_on_1",
    "cancer_20210122_slip2_area1_long_acq_MCF10A_tgfbeta_long_acq_blue_0.02909_green_0.0672_1",
    "cancer_20210122_slip1_area1_long_acq_MCF10A_tgfbeta_long_acq_blue_0.02909_green_0.0672_1",
    "cancer_20210314_slip8_area2_long_acq_MCF10A_37deg_long_acq_blue_0.06681_green_0.07975_1",
]


redo_trials_old = [
    "cancer_20210119_slip2_area1_long_acq_corr_long_acq_blue_0.0454_green_0.0671_1_overlay",
    "cancer_20210119_slip3_area2_long_acq_long_acq_blue_0.0454_green_0.0671_1",
    "cancer_20210119_slip4_area2_long_acq_long_acq_blue_0.0454_green_0.0671_1",
    "cancer_20210312_slip4_area3_long_acq_corr_MCF10A_36deg_long_acq_blue_0.0425_green_0.097_1",
    "cancer_20210312_slip4_area4_long_acq_corr_MCF10A_36deg_long_acq_blue_0.0425_green_0.097_1",
    "cancer_20210312_slip5_area3_long_acq_corr_corr_MCF10A_TGFB_36deg_long_acq_blue_0.0425_green_0.097_1",
    "cancer_20210313_slip1_area1_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.0506_green_0.097_1",
    "cancer_20210313_slip1_area2_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.0506_green_0.097_1",
    "cancer_20210313_slip1_area3_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.0506_green_0.097_1",
    "cancer_20210313_slip2_area1_long_acq_MCF10A_37deg_long_acq_blue_0.0506_green_0.097_1",
    "cancer_20210313_slip2_area3_long_acq_MCF10A_37deg_long_acq_blue_0.0506_green_0.097_1",
    "cancer_20210313_slip2_area4_long_acq_MCF10A_37deg_long_acq_blue_0.0506_green_0.097_1",
    "cancer_20210313_slip3_area3_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.0311_green_0.0489_1",
    "cancer_20210313_slip5_area2_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210314_slip1_area1_long_acq_MCF10A_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210314_slip3_area1_long_acq_MCF10A_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210314_slip4_area1_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210314_slip4_area3_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210314_slip5_area3_long_acq_MCF10A_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210314_slip6_area2_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210314_slip7_area2_long_acq_MCF10A_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210314_slip7_area3_long_acq_MCF10A_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210314_slip8_area1_long_acq_MCF10A_37deg_long_acq_blue_0.06681_green_0.07975_1",
    "cancer_20210314_slip8_area2_long_acq_MCF10A_37deg_long_acq_blue_0.06681_green_0.07975_1",
]


def make_all_grey_vids(
    top_dir,
    save_dir,
    initial_df,
    viewing_dir,
    thresh_idx,
    downsample=5,
    redo=False,
    QCd=False,
    onlymcf=False,
    HPC_num=None,
):
    df = pd.read_csv(initial_df)
    roi_df = pd.read_csv(Path(save_dir, "roi_df.csv"))

    qc_df = pd.read_csv(Path(save_dir, "good_detections.csv"))

    if QCd:
        namend = "_overlay_with_user_input"
    else:
        namend = "_overlay_no_user_input"

    for idx, data in enumerate(df.itertuples()):
        if HPC_num is not None:  # allows running in parallel on HPC
            if idx != HPC_num:
                continue

        trial_string = data.trial_string
        trial_save = Path(save_dir, "ratio_stacks", trial_string)
        print(trial_string)

        if data.use == "n":
            continue

        if (
            Path(viewing_dir, f"{data.trial_string}{namend}.tif").is_file()
            and not redo
            and trial_string not in redo_trials
        ):
            continue

        try:
            finish_at = int(data.finish_at) * 5
        except ValueError:
            finish_at = None

        seg = np.load(Path(trial_save, f"{trial_string}_seg.npy"))

        results = np.load(
            Path(trial_save, f"{trial_string}_event_properties.npy"), allow_pickle=True
        ).item()
        events = results["events"][thresh_idx]

        if QCd:
            print("doing all")
            bad_detections = [
                int(x.cell_id)
                for x in qc_df[qc_df.trial_string == data.trial_string].itertuples()
                if bool(x.correct) == False
            ]
            for x in bad_detections:
                if x not in events["excluded_circle_events"].keys():
                    events["excluded_events"][x] = events[x]
                    del events[x]

        if onlymcf and "MCF" not in data.expt:
            continue

        # only redo if there are events
        if (
            np.all([type(x) == str for x in events.keys()])
            and np.all([type(x) == str for x in events["excluded_events"].keys()])
            and True
        ):
            continue

        # if time.time() - os.path.getmtime(Path(viewing_dir,data.use, f'{data.trial_string}{namend}.tif')) < 10*60 and False:
        #    continue

        rat2 = np.load(Path(trial_save, f"{data.trial_string}_ratio_stack.npy"))[
            :finish_at
        ]
        rat2 = ndimage.gaussian_filter(rat2, (3, 2, 2))
        roi_overlay = make_roi_overlay(events, seg, rat2.shape)

        exclude_overlay = make_roi_overlay(events["excluded_events"], seg, rat2.shape)
        # exclude_circle_overlay = make_roi_overlay(events['excluded_circle_events'],seg,rat2.shape)
        downsample = 2
        alpha = 0.65

        # visualise circle exclusion
        circle_data = roi_df[roi_df.trial_string == data.trial_string]
        y, x = np.indices(seg.shape)

        y -= circle_data.circle_roi_center_y.values[0]
        x -= circle_data.circle_roi_center_x.values[0]

        r = np.sqrt(x ** 2 + y ** 2)
        exc = r > circle_data.circle_roi_radius.values[0]
        exc_outline = np.logical_xor(~exc, ndimage.binary_dilation(~exc, iterations=3))
        out_wh = np.where(exc_outline)

        # color balance
        cmin = np.percentile(rat2, 0.1)
        cmax = np.percentile(rat2, 99.9)
        rat2[np.where(rat2 < cmin)] = cmin
        rat2[np.where(rat2 > cmax)] = cmax
        rat2 = gf.norm(rat2)[::downsample]
        # alpha composite
        wh = np.where(roi_overlay[::downsample])
        rat2[wh] = rat2[wh] * (1 - alpha) + alpha

        wh = np.where(exclude_overlay[::downsample])
        rat2[wh] = rat2[wh] * (1 - alpha)

        rat2[:, out_wh[0], out_wh[1]] = 0
        tifffile.imsave(
            Path(viewing_dir, f"{data.trial_string}{namend}.tif"), gf.to_8_bit(rat2)
        )
