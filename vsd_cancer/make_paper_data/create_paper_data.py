#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:50:32 2020

@author: peter
"""

from pathlib import Path
import pandas as pd
from vsd_cancer.functions import cancer_functions as canf
import sys
import numpy as np
import os

redo = False

home = Path.home()
if "peq10" in str(home):
    HPC = True
    top_dir = Path(Path.home(), "firefly_link/cancer")
    df_str = "_HPC"
    HPC_num = int(sys.argv[1]) - 1  # allows running on HPC with data parallelism
    redo = bool(int(sys.argv[2]))
    print(f"Redoing: {redo}")
    yilins_computer = False
    yilin_save = False
elif "ys5320" in str(home):
    HPC = True
    top_dir = Path(Path.home(), "firefly_link/cancer")
    df_str = "_HPC"
    HPC_num = (
        int(sys.argv[1]) - 1
    )  # allows running on HPC with data parallelism with Yilin
    redo = bool(int(sys.argv[2]))
    print(f"Redoing: {redo}")
    yilins_computer = False
    yilin_save = False
elif os.name == "nt":
    HPC = False
    top_dir = Path("G:/")
    df_str = ""
    HPC_num = None
    yilin_save = True
    yilins_computer = True
    njobs = 6
elif "quickep" in str(home):
    HPC = False
    top_dir = Path("/Volumes/peq10/home/firefly_link/cancer")
    df_str = ""
    HPC_num = None
    yilin_save = False
    yilins_computer = False
    njobs = 10
else:
    HPC = False
    top_dir = Path("/home/peter/data/Firefly/cancer")
    df_str = ""
    HPC_num = None
    yilin_save = False
    yilins_computer = False
    njobs = 10


data_dir = Path(top_dir, "analysis", "full")
viewing_dir = Path(top_dir, "analysis", "full", "tif_viewing")


if not data_dir.is_dir():
    data_dir.mkdir()


print("Hello world")
initial_df = Path(top_dir, "analysis", "long_acqs_20220420_HPC_labelled_complete.csv")
redo_vid = False
if HPC:
    df_ = pd.read_csv(initial_df)
    print(f"Doing {df_.iloc[HPC_num].tif_file}")
    redo_vid = redo


print("Loading tif...")
if not yilins_computer:
    import load_all_long

    processed_df, failed_df = load_all_long.load_all_long(
        initial_df, data_dir, redo=False, HPC_num=HPC_num, use_SMR=False
    )
    # the failed only works when not redoing
    processed_df.to_csv(Path(data_dir, initial_df.stem + "_loaded_long.csv"))

    if not HPC:
        # look at failed
        failed_df = load_all_long.detect_failed(initial_df, data_dir)
        failed_df.to_csv(Path(data_dir, initial_df.stem + "_failed_loaded_long.csv"))

        # try to redo failed
        load_all_long.load_failed(
            Path(data_dir, initial_df.stem + "_failed_loaded_long.csv"), data_dir
        )

        # do no filt for wash in
        _, _ = load_all_long.load_all_long_washin(
            initial_df, data_dir, redo=False, HPC_num=HPC_num
        )


print("Segmenting...")
if redo:
    import segment_cellpose

    segment_cellpose.segment_cellpose(
        initial_df, data_dir, HPC_num=HPC_num, only_hand_rois=False
    )

print("Making overlays...")
if redo:
    import make_roi_overlays

    make_roi_overlays.make_all_overlay(
        initial_df, data_dir, Path(viewing_dir, "rois"), HPC_num=HPC_num
    )


print("Extracting time series...")
import make_all_t_courses

make_all_t_courses.make_all_tc(
    initial_df, data_dir, redo=redo, njobs=1, HPC_num=HPC_num, only_hand_rois=False
)


import make_all_cell_free_t_courses

make_all_cell_free_t_courses.make_all_cellfree_tc(
    initial_df, data_dir, redo=redo, HPC_num=HPC_num
)

print("Extracting FOV time series...")
import make_full_fov_t_courses

make_full_fov_t_courses.make_all_FOV_tc(
    initial_df, data_dir, redo=redo, HPC_num=HPC_num
)

import get_dead_cells

get_dead_cells.make_all_raw_tc(
    initial_df, data_dir, redo=redo, njobs=1, HPC_num=HPC_num
)


print("Getting mean brightnesses")
import get_all_brightness

get_all_brightness.get_mean_brightness(initial_df, data_dir, HPC_num=HPC_num)
print(f"HPC_num = {HPC_num}")
if True:
    print("Defining circle exclusion")
    import define_circle_rois

    define_circle_rois.define_circle_rois(
        top_dir,
        initial_df,
        data_dir,
        radius=220,
        center=(246, 256),
        HPC_num=HPC_num,
        redo=redo,
    )

    import apply_circle_rois

    print("Applying circle exclusion")
    apply_circle_rois.apply_circle_exclusion(
        top_dir, data_dir, initial_df, HPC_num=HPC_num
    )
    print(f"HPC_num = {HPC_num}")

if False:
    print("Getting LED calibration...")
    import get_LED_calibration

    get_LED_calibration.get_LED_calibration(top_dir, data_dir)

    import get_led_powers

    get_led_powers.get_all_powers(initial_df, data_dir)

print("Detecting events...")
import get_events

print(f"HPC_num = {HPC_num}")
if redo:
    get_events.get_measure_events(
        initial_df,
        data_dir,
        thresh_range=np.arange(2, 4.5, 0.5),
        surrounds_z=10,
        exclude_first=400,
        tc_type="median",
        exclude_circle=True,
        yilin_save=yilin_save,
        HPC_num=HPC_num,
    )


thresh_idx = 1
print("Making videos...")
import make_corr_grey_vids

print(f"HPC_num = {HPC_num}")
make_corr_grey_vids.make_all_grey_vids(
    top_dir,
    data_dir,
    initial_df,
    Path(viewing_dir, "final_paper_before_user_input"),
    thresh_idx,
    redo=redo_vid,
    QCd=False,
    HPC_num=HPC_num,
)


print("Getting user input for good detections")
if HPC_num is None:
    import get_all_good_detections

    get_all_good_detections.get_user_event_input(
        initial_df,
        data_dir,
        Path(viewing_dir, "final_paper_before_user_input"),
        thresh_idx,
        redo=False,
    )

    print("Exporting events...")
    import export_events

    export_events.export_events(
        initial_df, data_dir, thresh_idx, min_ttx_amp=0, amp_threshold=None
    )

    print("Making videos...")
    import make_corr_grey_vids

    make_corr_grey_vids.make_all_grey_vids(
        top_dir,
        data_dir,
        initial_df,
        Path(viewing_dir, "final_paper_after_user_input"),
        thresh_idx,
        redo=False,
        QCd=True,
        onlymcf=False,
    )

    import make_spike_trains

    make_spike_trains.export_spike_trains(data_dir, T=0.2, only_neg=True)

    import bootstrap_correlation_analysis

    bootstrap_correlation_analysis.calculate_corrs(top_dir, data_dir, redo=True)

    import make_paper_figures.make_all_figures

    print("Finished successfully")
else:
    # Temporary to get first pass answers mark alll detections as quality controlled

    import fake_get_all_good_detections

    fake_get_all_good_detections.get_user_event_input(
        initial_df,
        data_dir,
        Path(viewing_dir, "final_paper_before_user_input"),
        thresh_idx,
        redo=False,
    )

    print("Exporting events...")
    import export_events

    export_events.export_events(
        initial_df, data_dir, thresh_idx, min_ttx_amp=0, amp_threshold=None
    )
