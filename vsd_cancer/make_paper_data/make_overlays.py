#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 17:26:21 2021

@author: peter
"""
import numpy as np
from pathlib import Path
import pandas as pd

import scipy.ndimage as ndimage
import scipy.signal as signal

import matplotlib.cm
import tifffile
import time

import f.plotting_functions as pf
import f.general_functions as gf

from vsd_cancer.functions import cancer_functions as canf


def make_overlay(
    arr, stack, mask, cmap=matplotlib.cm.hot, alpha_top=0.7, percent=5, contrast=[0, 0]
):
    # mask

    overlay = cmap(arr)
    overlay[..., -1] *= (~mask).astype(int) * alpha_top
    underlay = matplotlib.cm.Greys_r(stack)

    # this does general alpha composite - more efficient below due to specifics
    # final = pf.alpha_composite(gf.to_8_bit(overlay), underlay)

    overlay[..., :-1] *= overlay[..., -1][
        ..., None
    ]  # premultiply for alpha compositing
    final = np.ones_like(underlay)
    final[..., :-1] = overlay[..., :-1] + underlay[..., :-1] * (
        1 - overlay[..., -1][..., None]
    )
    return (final * 255).astype(np.uint8)


def chunk_overlay(
    arr,
    norm_stack,
    chunk_size,
    cmap=matplotlib.cm.hot,
    alpha_top=0.7,
    percent=5,
    contrast=[0, 0],
):
    res = np.zeros(arr.shape + (4,), dtype=np.uint8)
    n_chunks, rem = np.divmod(arr.shape[0], chunk_size)
    mask = np.logical_and(
        arr < np.percentile(arr, 100 - percent), arr > np.percentile(arr, percent)
    )
    # apply contrast adjustment
    ma, mi = np.percentile(arr, 100 - contrast[0]), np.percentile(arr, contrast[0])
    arr[arr > ma] = ma
    arr[arr < mi] = mi
    arr = gf.norm(arr)
    ma2, mi2 = (
        np.percentile(norm_stack, 100 - contrast[1]),
        np.percentile(norm_stack, contrast[1]),
    )  #
    norm_stack[norm_stack > ma2] = ma2
    norm_stack[norm_stack < mi2] = mi2
    norm_stack = gf.norm(norm_stack)
    for i in range(n_chunks):
        res[i * chunk_size : (i + 1) * chunk_size, ...] = make_overlay(
            arr[i * chunk_size : (i + 1) * chunk_size, ...],
            norm_stack[i * chunk_size : (i + 1) * chunk_size, ...],
            mask[i * chunk_size : (i + 1) * chunk_size, ...],
            cmap=cmap,
            alpha_top=alpha_top,
            percent=percent,
            contrast=contrast,
        )

    if rem != 0:
        res[-rem:, ...] = make_overlay(
            arr[-rem:, ...],
            norm_stack[-rem:, ...],
            mask[-rem:, ...],
            cmap=cmap,
            alpha_top=alpha_top,
            percent=percent,
            contrast=contrast,
        )
    return res


def make_roi_overlay(events_dict, seg, sz):
    overlay = np.zeros(sz, dtype=int)
    for idx in events_dict.keys():
        if type(idx) == str:
            continue
        for idx2 in range(events_dict[idx].shape[-1]):
            ids = events_dict[idx][:, idx2]
            mask = (seg == idx + 1).astype(int)
            outline = np.logical_xor(
                mask, ndimage.binary_dilation(mask, iterations=2)
            ).astype(int)
            overlay[ids[0] : ids[1], ...] += outline

    overlay = overlay > 0

    return overlay


top_dir = Path("/home/peter/data/Firefly/cancer")
df_str = ""
HPC_num = None


save_dir = Path(top_dir, "analysis", "full")
viewing_dir = Path(top_dir, "analysis", "full", "tif_viewing", "videos")
initial_df = Path(
    top_dir, "analysis", f"long_acqs_20210428_experiments_correct{df_str}.csv"
)

df = pd.read_csv(initial_df)


def make_overlay_events(rat, stack, seg, evs=None, downsample=5):
    rat = rat[:, 2:-2, 2:-2]
    rat = ndimage.filters.gaussian_filter(rat, (3, 2, 2))
    rat = np.pad(rat, ((0, 0), (2, 2), (2, 2)), mode="edge")

    ovs = []
    for e in evs:
        ovs.append(make_roi_overlay(e, seg, rat.shape)[::downsample, ...])

    display = chunk_overlay(
        rat[::downsample],
        stack[::downsample],
        5000,
        cmap=matplotlib.cm.Spectral,
        alpha_top=0.2,
        percent=50,
        contrast=[0.5, 0.1],
    )

    colors = np.array([[255, 0, 0, 255], [0, 127, 0, 127], [0, 0, 127, 127]])

    for idx, o in enumerate(ovs):
        wh = np.where(o)
        display[wh[0], wh[1], wh[2], :] = colors[idx]

    return display


downsample = 5

for idx, data in enumerate(df[::-1].itertuples()):

    t0 = time.time()

    trial_string = data.trial_string
    trial_save = Path(save_dir, "ratio_stacks", trial_string)
    print(trial_string)
    if Path(viewing_dir, f"{data.trial_string}_overlay_2.tif").is_file() and True:
        continue

    if data.use == "n":
        continue
    # if trial_string != 'cancer_20201203_slip1_area2_long_acq_corr_corr_long_acqu_blue_0.0551_green_0.0832_heated_to_37_1':
    #    continue

    # if trial_string != 'cancer_20201215_slip2_area1_long_acq_corr_long_acq_blue_0.0296_green_0.0765_heated_to_37_1':
    #    continue

    try:
        finish_at = int(data.finish_at) * 5
    except ValueError:
        finish_at = None

    rat2 = np.load(Path(trial_save, f"{data.trial_string}_ratio_stack.npy"))[:finish_at]
    rat2 = ndimage.gaussian_filter(rat2, (3, 2, 2))

    tc = np.load(Path(trial_save, f"{trial_string}_all_tcs.npy"))[:finish_at]
    std = np.load(Path(trial_save, f"{trial_string}_all_stds.npy"))[:finish_at]
    tc -= tc.mean(-1)[:, None] - 1

    seg = np.load(Path(trial_save, f"{trial_string}_seg.npy"))
    filt_params = {"type": "TV", "TV_weight": 0.01, "gaussian_sigma": 3}

    # exclude_dict = np.load(Path(trial_save,f'{trial_string}_processed_exclusions.npy'),allow_pickle = True).item()
    # add exclusion
    # excluded_tc = canf.apply_exclusion(exclude_dict,tc)
    masks = canf.lab2masks(seg)

    surround_tc = np.load(Path(trial_save, f"{trial_string}_all_surround_tcs.npy"))[
        :finish_at
    ]
    surround_std = np.load(Path(trial_save, f"{trial_string}_all_surround_stds.npy"))[
        :finish_at
    ]
    surround_tc -= np.mean(surround_tc, -1)[:, None] - 1
    excluded_circle = np.load(
        Path(trial_save, f"{trial_string}_circle_excluded_rois.npy")
    )

    excluded_die = np.load(Path(trial_save, f"{trial_string}_excluded_dead_rois.npy"))

    events = canf.get_events_exclude_surround_events(
        tc,
        std,
        surround_tc,
        surround_std,
        z_score=2.5,
        surround_z=5,
        exclude_first=0,
        excluded_circle=excluded_circle,
        excluded_dead=excluded_die,
    )

    roi_overlay = make_roi_overlay(events, seg, rat2.shape)
    exclude_overlay = make_roi_overlay(events["excluded_events"], seg, rat2.shape)
    exclude_circle_overlay = make_roi_overlay(
        events["excluded_circle_events"], seg, rat2.shape
    )

    stack = tifffile.imread(data.tif_file)[::2, ...]

    display = make_overlay_events(
        rat2,
        stack,
        seg,
        evs=[events, events["excluded_events"], events["excluded_circle_events"]],
    )

    tifffile.imsave(Path(viewing_dir, f"{data.trial_string}_overlay_2.tif"), display)

    print(time.time() - t0)
