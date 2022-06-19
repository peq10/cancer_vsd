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

redo_trials = []


def get_user_event_input(initial_df, save_dir, viewing_dir, thresh_idx, redo=True):
    """
    this is janky but works
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
                    vid = tifffile.imread(vidpath)

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
                            trial_save,
                            f"{trial_string}_NEW_good_detection_cell_{ce}.npy",
                        )
                        # also make a small video around cell
                        if (
                            ffiile.is_file()
                            and not redo
                            and data.trial_string not in redo_trials
                        ):
                            detection_real = np.load(ffiile)
                        else:
                            # raise ValueError('Have to do ONE PER DETECTION')
                            event_vid = []
                            for timeee in times[idxxx].T:
                                event_vid.append(
                                    vid[
                                        max(timeee[0] // 2 - 50, 0) : timeee[1] // 2
                                        + 50,
                                        :,
                                        :,
                                    ]
                                )

                            event_vid = np.concatenate(event_vid)

                            # label events with red spot in top left
                            for evv in times[idxxx].T:
                                t0 = times[idxxx][0, 0]
                                event_vid[evv[0] - t0 : evv[1] - t0, :10, :10] = 0

                                # label the cell location
                            rad = 20
                            r = np.sqrt(
                                np.sum(
                                    (
                                        np.indices(event_vid.shape[1:])
                                        - locs[idxxx][:, None, None]
                                    )
                                    ** 2,
                                    0,
                                )
                            )
                            r = np.logical_and(r < rad + 3, r > rad)
                            rwh = np.where(r)

                            ii = 0
                            windowname = f"{trial_string} Cell {ce}"
                            view_window = cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
                            cv2.setWindowProperty(
                                windowname,
                                cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN,
                            )
                            cv2.setWindowProperty(
                                windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                            )
                            cv2.resizeWindow(windowname, 2000, 2000)
                            while True:

                                # Display the resulting frame

                                fr = cv2.cvtColor(
                                    event_vid[ii % event_vid.shape[0]],
                                    cv2.COLOR_GRAY2RGB,
                                )
                                fr[rwh[0], rwh[1], :] = [0, 0, 255]
                                cv2.imshow(windowname, fr)

                                # Press Q on keyboard to  exit
                                if cv2.waitKey(10) & 0xFF == ord("y"):
                                    detection_real = True
                                    break
                                elif cv2.waitKey(10) & 0xFF == ord("n"):
                                    detection_real = False
                                    break

                                ii += 1

                            cv2.destroyAllWindows()

                            np.save(ffiile, detection_real)
                            print(f"Done {ffiile}")

                        detected_frame.loc[detections, "correct"] = detection_real
                        detections += 1
    detected_frame.to_csv(
        Path(
            save_dir,
            f"{initial_df.stem}_intermediate_files",
            f"{initial_df.stem}_good_detections.csv",
        )
    )


if __name__ == "__main__":
    top_dir = Path(Path.home(), "firefly_link/cancer")
    df_str = ""
    save_dir = Path(top_dir, "analysis", "full")
    viewing_dir = Path(top_dir, "analysis", "full", "tif_viewing")
    initial_df = Path(
        top_dir,
        "analysis",
        "correct_dataframes",
        "long_acqs_20220420_HPC_labelled_complete.csv",
    )
    data_dir = Path(top_dir, "analysis", "full")
    viewing_dir = Path(
        top_dir, "analysis", "full", "tif_viewing", "final_paper_before_user_input"
    )
    thresh_idx = 1
    get_user_event_input(initial_df, save_dir, viewing_dir, thresh_idx, redo=False)
