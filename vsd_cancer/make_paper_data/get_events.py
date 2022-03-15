#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:09:07 2021

@author: peter
"""
# a script to get all the events

import numpy as np

import pandas as pd
from pathlib import Path

from vsd_cancer.functions import cancer_functions as canf

import matplotlib.pyplot as plt


def get_dead_cells(raw_tc):
    raw_tc /= raw_tc[:, 0][:, None]
    dead = np.where(raw_tc[:, -1] > 1.25)[0]
    return dead


def get_measure_events(
    initial_df,
    save_dir,
    thresh_range=np.arange(2, 4.5, 0.5),
    surrounds_z=10,
    exclude_first=0,
    tc_type="median",
    exclude_circle=False,
    overlap=0.7,
    simultaneous=5,
    MCF_overlap=0.3,
    MCF_simultaneous=3,
    yilin_save=False,
):

    df = pd.read_csv(initial_df)
    for idx, data in enumerate(df.itertuples()):
        # if idx != 34:
        #    continue

        trial_string = data.trial_string
        print(trial_string)
        trial_save = Path(save_dir, "ratio_stacks", trial_string)

        if tc_type == "median":
            tc = np.load(Path(trial_save, f"{trial_string}_all_eroded_median_tcs.npy"))
        else:
            tc = np.load(Path(trial_save, f"{trial_string}_all_eroded_tcs.npy"))

        tc -= np.mean(tc, -1)[:, None] - 1

        std = np.load(Path(trial_save, f"{trial_string}_all_stds.npy"))
        surround_std = np.load(
            Path(trial_save, f"{trial_string}_all_surround_stds.npy")
        )

        if exclude_circle:
            excluded_circle = np.load(
                Path(trial_save, f"{trial_string}_circle_excluded_rois.npy")
            )
        else:
            excluded_circle = None

        # also get circle exclusions
        surround_tc = np.load(Path(trial_save, f"{trial_string}_all_surround_tcs.npy"))
        # remove any surround offsets
        surround_tc -= np.mean(surround_tc, -1)[:, None] - 1

        raw_tc = np.load(Path(trial_save, f"{trial_string}_raw_tc.npy"))

        if not np.isnan(data.finish_at):
            observe_to = int(data.finish_at) * 5
            tc = tc[:, :observe_to]
            std = std[:, :observe_to]
            surround_tc = surround_tc[:observe_to]
            surround_std = surround_std[:, :observe_to]
            raw_tc = raw_tc[:, :observe_to]

        dead_idx = get_dead_cells(raw_tc)
        np.save(Path(trial_save, f"{trial_string}_excluded_dead_rois.npy"), dead_idx)

        all_events = []
        all_observation = []
        for detection_thresh in thresh_range:

            if "MCF" in data.expt:
                events = canf.get_events_exclude_simultaneous_events(
                    tc,
                    std,
                    z_score=detection_thresh,
                    max_events=MCF_simultaneous,
                    overlap=MCF_overlap,
                    exclude_first=exclude_first,
                    excluded_circle=excluded_circle,
                    excluded_dead=dead_idx,
                )

            else:
                events = canf.get_events_exclude_simultaneous_events(
                    tc,
                    std,
                    z_score=detection_thresh,
                    max_events=simultaneous,
                    overlap=overlap,
                    exclude_first=exclude_first,
                    excluded_circle=excluded_circle,
                    excluded_dead=dead_idx,
                )

            event_with_props = canf.get_event_properties(events, use_filt=False)

            all_events.append(event_with_props)
            all_observation.append(canf.get_observation_length(events))

        detect_params = {
            "thresh_range": thresh_range,
            "surrounds_thresh": surrounds_z,
            "exclude_first": exclude_first,
        }

        if exclude_circle == False:
            result_dict = {
                "n_cells": tc.shape[0],
                "events": all_events,
                "observation_length": all_observation,
                "excluded_circle": excluded_circle,
                "detection_params": detect_params,
            }
        else:
            result_dict = {
                "n_cells": tc.shape[0] - len(excluded_circle),
                "events": all_events,
                "observation_length": all_observation,
                "excluded_circle": excluded_circle,
                "detection_params": detect_params,
            }

        # all_props = np.concatenate([event_props[p] for p in event_props.keys() if 'props' in str(p)])
        if not yilin_save:
            print("Overwriting previous event props")
            np.save(
                Path(trial_save, f"{trial_string}_event_properties.npy"), result_dict
            )

        else:
            print("Not overwriting previous event props - saving for yilin")
            ypath = Path(
                trial_save,
                "../../yilin_event_props",
                f"{trial_string}_event_properties_yilin_copy.npy",
            )
            print(ypath.absolute())
            np.save(ypath, result_dict)
