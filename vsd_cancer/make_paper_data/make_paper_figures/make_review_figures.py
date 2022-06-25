#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 11:08:08 2022

@author: quickep
"""

from pathlib import Path

# from vsd_cancer.functions import cancer_functions as canf

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import f.plotting_functions as pf

import matplotlib.cm
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import seaborn as sns

import statsmodels.stats.multitest
import scipy.stats
import itertools


def make_figures(initial_df, save_dir, figure_dir, filetype=".png"):
    figsave = Path(figure_dir, "cell_lines_figure")
    if not figsave.is_dir():
        figsave.mkdir()

    df = pd.read_csv(
        Path(
            save_dir,
            "20220423_original_and_review_intermediate_files/20220423_original_and_review_non_ttx_active_df_by_cell.csv",
        )
    )
    df2 = pd.read_csv(
        Path(
            save_dir,
            "20220423_original_and_review_intermediate_files/20220423_original_and_review_TTX_active_df_by_cell.csv",
        )
    )
    df2 = df2[df2.stage == "pre"]

    df3 = pd.read_csv(
        Path(
            save_dir,
            "long_acqs_20220420_HPC_labelled_complete_intermediate_files/long_acqs_20220420_HPC_labelled_complete_non_ttx_active_df_by_cell.csv",
        )
    )

    df = pd.concat([df, df2, df3])

    T = 0.2

    df["exp_stage"] = df.expt + "_" + df.stage
    df["day_slip"] = df.day.astype(str) + "_" + df.slip.astype(str)

    df["neg_event_rate"] = (df["n_neg_events"]) / (df["obs_length"] * T)

    MDA_keys = [
        "standard_none",
        "TTX_10um_pre",
        "TTX_10um_washout_pre",
        "TTX_1um_pre",
        "L231_none",
    ]

    df.exp_stage = [x if x not in MDA_keys else "L231_none" for x in df.exp_stage]

    lines = list(np.unique(df.exp_stage))

    lines = [l for l in lines if "TGFB" not in l]

    results_trial = []

    results_slip = []

    for l in lines:

        results_trial.append(
            df[df.exp_stage == l].groupby("trial").mean()["neg_event_rate"].to_numpy()
        )
        results_slip.append(
            df[df.exp_stage == l]
            .groupby("day_slip")
            .mean()["neg_event_rate"]
            .to_numpy()
        )

    # deal with the annoying thing where sometimes the line has L in front of it
    lines2 = [x[1:] if x[0] == "L" else x for x in lines]

    un_cell_lines = list(set(lines2))
    order = ["MCF10A"]

    results_trial_corr = []

    names = ["_".join(x.split("_")[:-1]) for x in un_cell_lines]

    for l in un_cell_lines:

        results_trial_corr.append([])
        for l2, res in zip(lines2, results_trial):
            if l2 == l:
                results_trial_corr[-1].extend(res)

    fig, ax = plt.subplots()
    sns.swarmplot(data=[np.array(x) * 1000 for x in results_trial_corr], ax=ax)

    ax.set_xticks(range(len(un_cell_lines)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_xlabel("Cell Line")
    ax.set_ylabel("Negative event rate (events/cell/1000 s)")
    fig.savefig(
        Path(
            figsave,
            f"all_cells_plot{filetype}",
        ),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )

    kruskal_p = scipy.stats.kruskal(*results_trial_corr)

    MDA_res = results_trial_corr[un_cell_lines.index("231_none")]
    MCF_res = results_trial_corr[un_cell_lines.index("MCF10A_none")]

    ps_MCF = []
    ps_MDA = []

    names = []
    for l, res in zip(un_cell_lines, results_trial_corr):
        if l != "231_none":
            p_mda = scipy.stats.mannwhitneyu(res, MDA_res)
            ps_MDA.append((l, p_mda.pvalue))

        if l != "MCF10A_none":
            p_mcf = scipy.stats.mannwhitneyu(res, MCF_res)
            ps_MCF.append((l, p_mcf.pvalue))
            names.append(l)

    corrected_10a = statsmodels.stats.multitest.multipletests(
        [x[1] for x in ps_MCF], method="fdr_bh"
    )

    ps_MCF_corrected = [
        (name.split("_")[0], corr_p) for name, corr_p in zip(names, corrected_10a[1])
    ]
    print(ps_MCF_corrected)


if __name__ == "__main__":

    top_dir = Path("/home/peter/bigdata/Firefly/cancer/")
    save_dir = Path(top_dir, "analysis", "full")
    figure_dir = Path("/home/peter/Dropbox/Papers/cancer/reviews")
    initial_df = Path(top_dir, "analysis", "long_acqs_20210428_experiments_correct.csv")
    make_figures(initial_df, save_dir, figure_dir, filetype=".png")
