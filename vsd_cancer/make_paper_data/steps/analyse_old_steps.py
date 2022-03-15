#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:13:08 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import cancer_functions as canf
import f.general_functions as gf

import scipy.stats

df = pd.read_csv("/home/peter/data/Firefly/cancer/analysis/old_steps.csv")


mean_fs = []
mean_vs = []
mean_is = []
mean_rs = []
fits = []
sens = []

for idx, data in enumerate(df.itertuples()):

    ephys_file = [str(f) for f in Path(data.ephys_loc).glob("*.smr")][0]
    im_dir = Path(data.directory)

    f = [f for f in im_dir.glob("./**/*.tif")][0]

    trial_string = "_".join(f.parts[5:-1])

    df.loc[data.Index, "trial_string"] = trial_string

    trial_save = Path(
        "/home/peter/data/Firefly/cancer/analysis/full",
        "steps_analysis/data",
        trial_string,
    )
    if not trial_save.is_dir():
        trial_save.mkdir(parents=True)
    print(trial_string)

    ephys_dict, stacks = canf.get_steps_image_ephys(im_dir, ephys_file)

    # first blue is bad
    stacks[:, 0, ...] = stacks[:, 2, ...]

    interped_stack = canf.process_ratio_stacks(stacks)

    _, roi = gf.read_roi_file(
        Path(
            "/home/peter/data/Firefly/cancer/analysis/full",
            "steps_analysis/rois",
            f"{trial_string}.roi",
        ),
        im_dims=interped_stack.shape[-2:],
    )

    # now get the time courses
    t_courses = gf.t_course_from_roi(interped_stack, roi)

    # use linear fit for bleaching
    sta = np.mean(t_courses[..., :5], -1)
    sto = np.mean(t_courses[..., -5:], -1)
    m = (sto - sta) / t_courses.shape[-1]

    lin_fit = (
        np.arange(t_courses.shape[-1])[None, None, :] * m[:, :, None] + sta[:, :, None]
    )

    offset = 90 * 16

    df_t = (t_courses - lin_fit) / (lin_fit - offset)

    np.save(Path(trial_save, f"{trial_string}_df_tc.npy"), df_t)

    stim_locs = np.array([42, 86])

    mean_f = np.mean(df_t[..., stim_locs[0] : stim_locs[1]], -1)

    mean_fs.append(mean_f)

    dr_t = (df_t[:, 0, :] + 1) / (df_t[:, 1, :] + 1)

    mean_r = np.mean(dr_t[..., stim_locs[0] : stim_locs[1]], -1)
    mean_rs.append(mean_r)

    vm = ephys_dict["vcVm_sliced"]
    im = ephys_dict["vcIm_sliced"]
    v_locs = np.round((stim_locs / t_courses.shape[-1]) * vm.shape[-1]).astype(int)

    mean_v = np.mean(vm[:, v_locs[0] : v_locs[1]], -1)
    mean_vs.append(mean_v)

    mean_i = np.mean(im[:, v_locs[0] : v_locs[1]], -1)
    mean_is.append(mean_i)

    np.save(Path(trial_save, f"{trial_string}_vm.npy"), vm)
    np.save(Path(trial_save, f"{trial_string}_im.npy"), ephys_dict["vcIm_sliced"])

    fit_blue = scipy.stats.linregress(mean_v, mean_f[:, 0])
    fit_green = scipy.stats.linregress(mean_v, mean_f[:, 1])
    fit_rat = scipy.stats.linregress(mean_v, mean_r)
    fit_ephys = scipy.stats.linregress(mean_i, mean_v)

    fits.append([fit_blue, fit_green, fit_rat])

    sens.append([fit_blue.slope, fit_green.slope, fit_rat.slope])

sens = np.array(sens)


df.to_csv("/home/peter/data/Firefly/cancer/analysis/old_steps.csv")
