#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:45:49 2020

@author: peter
"""

import numpy as np
from pathlib import Path
import pandas as pd


from vsd_cancer.functions import cancer_functions as canf


def make_all_cellfree_tc(df_file, save_dir, redo=True, HPC_num=None):
    df = pd.read_csv(df_file)

    if redo or HPC_num is not None:
        redo_from = 0
    else:
        try:
            redo_from = np.load(
                Path(
                    save_dir,
                    f"{df_file.stem}_intermediate_files",
                    f"{df_file.stem}_redo_from_make_all_cellfree_tc.npy",
                )
            )
            print(f"{len(df) - redo_from} to do")
        except FileNotFoundError:
            redo_from = 0

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
            if Path(trial_save, f"{trial_string}_all_cellfree_tc.npy").is_file():
                continue

        seg = np.load(Path(trial_save, f"{trial_string}_seg.npy"))
        mask = seg == 0

        stack = np.load(Path(trial_save, f"{trial_string}_ratio_stack.npy")).astype(
            np.float64
        )

        tc = canf.t_course_from_roi(stack, mask)
        std = canf.std_t_course_from_roi(stack, mask, True)

        tc = np.array(tc)
        tc -= tc.mean(-1) - 1

        np.save(Path(trial_save, f"{trial_string}_cellfree_tc.npy"), tc)
        np.save(Path(trial_save, f"{trial_string}_cellfree_std.npy"), std)

        print(f"Saved {trial_string}")
        redo_from += 1
        np.save(
            Path(
                save_dir,
                f"{df_file.stem}_intermediate_files",
                f"{df_file.stem}_redo_from_make_all_cellfree_tc.npy",
            ),
            redo_from,
        )
