#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:45:49 2020

@author: peter
"""

import numpy as np
from pathlib import Path
import pandas as pd


def make_all_FOV_tc(df_file, save_dir, redo=True, HPC_num=None):
    df = pd.read_csv(df_file)

    if redo or HPC_num is not None:
        redo_from = 0
    else:
        redo_from = np.load(
            Path(
                save_dir,
                f"{df_file.stem}_intermediate_files",
                f"{df_file.stem}_redo_from_make_full_fov_tc.npy",
            )
        )
        print(f"{len(df) - redo_from} to do")

    for idx, data in enumerate(df.itertuples()):
        if HPC_num is not None:  # allows running in parallel on HPC
            if idx != HPC_num:
                continue

        parts = Path(data.tif_file).parts
        trial_string = "_".join(parts[parts.index("cancer") : -1])
        trial_save = Path(save_dir, "ratio_stacks", trial_string)

        if not redo and HPC_num is None:
            if idx < redo_from:
                continue
        elif not redo and HPC_num is not None:
            if Path(trial_save, f"{trial_string}_full_fov_tc.npy").is_file():
                print("Skipping FOV tc")
                continue

        stack = np.load(Path(trial_save, f"{trial_string}_ratio_stack.npy")).astype(
            np.float64
        )

        tc = np.mean(stack, axis=(-2, -1))
        std = np.std(stack, axis=(-2, -1)) / np.sqrt(np.multiply(*stack.shape[-2:]))

        tc = np.array(tc)
        tc -= tc.mean(-1) - 1

        np.save(Path(trial_save, f"{trial_string}_full_fov_tc.npy"), tc)
        np.save(Path(trial_save, f"{trial_string}_full_fov_std.npy"), std)

        print(f"Saved {trial_string} FOV TC")
        redo_from += 1
        np.save(
            Path(
                save_dir,
                f"{df_file.stem}_intermediate_files",
                f"{df_file.stem}_redo_from_make_full_fov_tc.npy",
            ),
            redo_from,
        )
