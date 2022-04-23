#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:31:26 2021

@author: peter
"""

import numpy as np

import pandas as pd
from pathlib import Path

import scipy.ndimage as ndimage

import tifffile
import pdb
import cv2


def get_user_event_input(initial_df, save_dir, viewing_dir, thresh_idx, redo=True):
    """
    this is a script that fakes a user inputting all the detections as good so I can get a first pass at the results without manual QC
    """

    df = pd.read_csv(initial_df)
    df = df[df.use != "n"]

    use = [True if "washin" not in x else False for x in df.expt]
    df = df[use]

    trial_string = df.iloc[0].trial_string

    detected_frame = pd.DataFrame()
    detections = 0
    use_idx = thresh_idx

    for idx, data in enumerate(df.itertuples()):

        trial_string = data.trial_string

        # print(trial_string)
        trial_save = Path(save_dir, "ratio_stacks", trial_string)

        results = np.load(
            Path(trial_save, f"{trial_string}_event_properties.npy"), allow_pickle=True
        ).item()
        seg = np.load(Path(trial_save, f"{trial_string}_seg.npy"))
        cell_ids = np.arange(results["events"][0]["tc_filt"].shape[0])

        if results["excluded_circle"] is not None:
            cell_ids = [x for x in cell_ids if x not in results["excluded_circle"]]

        # if trial_string == 'cancer_20210314_slip2_area3_long_acq_MCF10A_TGFB_37deg_long_acq_blue_0.06681_green_0.07975_1':
        #    pdb.set_trace()

        for idx, thresh_level_dict in enumerate(results["events"]):

            if idx != use_idx:
                continue

            event_props = results["events"][idx]["event_props"]

            sum_current = [
                np.sum(np.abs(event_props[x][:, -1])) if x in event_props.keys() else 0
                for x in cell_ids
            ]

            # manually check finds
            if idx == use_idx:
                if np.any(np.array(sum_current) != 0):
                    vidpath = [
                        x for x in Path(viewing_dir).glob(f"./**/*{trial_string}*")
                    ][0]

                    active_cells = [x for x in results["events"][idx] if type(x) != str]
                    locs = np.round(
                        [ndimage.center_of_mass(seg == x + 1) for x in active_cells]
                    ).astype(int)
                    times = [results["events"][idx][x] for x in active_cells]
                    for idxxx, ce in enumerate(active_cells):
                        detected_frame.loc[detections, "trial_string"] = trial_string
                        detected_frame.loc[detections, "cell_id"] = ce
                        detected_frame.loc[detections, "loc"] = str(locs[idxxx])
                        detected_frame.loc[detections, "starts"] = str(
                            times[idxxx][0, :] / 2
                        )
                        ffiile = Path(
                            trial_save, f"{trial_string}_good_detection_cell_{ce}.npy"
                        )
                        # also make a small video around cell
                        if (
                            Path(
                                trial_save,
                                f"{trial_string}_good_detection_cell_{ce}.npy",
                            ).is_file()
                            and not redo
                        ):
                            detection_real = np.load(ffiile)
                        else:

                            np.save(ffiile, True)
                            print(f"Done {ffiile}")

                        detected_frame.loc[detections, "correct"] = True
                        detections += 1
    detected_frame.to_csv(
        Path(
            save_dir,
            f"{initial_df.stem}_intermediate_files",
            f"{initial_df.stem}_good_detections.csv",
        )
    )
