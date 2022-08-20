#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:56:27 2021

@author: peter
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 18:25:53 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from vsd_cancer.functions import cancer_functions as canf
import f.general_functions as gf
import f.plotting_functions as pf
import scipy.stats
import tifffile
import scipy.ndimage as ndimage


import seaborn as sns


def make_figures(figure_dir, filetype=".png"):

    figsave = Path(figure_dir, "patch_figure")
    if not figsave.is_dir():
        figsave.mkdir()

    df = pd.read_csv("/home/peter/data/Firefly/cancer/analysis/old_steps.csv")

    df["old"] = True

    df2 = pd.read_csv(
        "/home/peter/data/Firefly/cancer/analysis/steps_20201230_sorted.csv"
    )
    df2 = df2[df2.run == 0]

    df2["old"] = False

    df = df.append(df2)

    mean_fs = []
    mean_vs = []
    mean_is = []
    mean_rs = []
    fits = []
    sens = []

    example = "cancer_20201113_slip1_cell1_steps_steps_with_emission_ratio_steps_green_0.125_blue_0.206_1"
    T = 1 / 50

    for idx, data in enumerate(df.itertuples()):

        if data.old:
            trial_string = data.trial_string
        else:
            s = data.tif_file
            trial_string = "_".join(Path(s).parts[Path(s).parts.index("cancer") : -1])
        trial_save = Path(
            "/home/peter/data/Firefly/cancer/analysis/full",
            "steps_analysis/data",
            trial_string,
        )

        df_t = np.load(Path(trial_save, f"{trial_string}_df_tc.npy"))
        vm = np.load(Path(trial_save, f"{trial_string}_vm.npy"))
        im = np.load(Path(trial_save, f"{trial_string}_im.npy"))

        if data.old:
            stim_locs = np.array([42, 86])
        else:
            stim_locs = np.array([25, 49])

        mean_f = np.mean(df_t[..., stim_locs[0] : stim_locs[1]], -1)
        mean_fs.append(mean_f)

        # plt.plot(df_t[:,1,:].T)*100

        dr_t = (df_t[:, 0, :] + 1) / (df_t[:, 1, :] + 1)

        mean_r = np.mean(dr_t[..., stim_locs[0] : stim_locs[1]], -1)
        mean_rs.append(mean_r)

        # plt.plot(dr_t.T)
        # plt.show()
        # print(trial_string)

        v_locs = np.round((stim_locs / df_t.shape[-1]) * vm.shape[-1]).astype(int)

        mean_v = np.mean(vm[:, v_locs[0] : v_locs[1]], -1)
        mean_vs.append(mean_v)

        mean_i = np.mean(im[:, v_locs[0] : v_locs[1]], -1)
        mean_is.append(mean_i)

        fit_blue = scipy.stats.linregress(mean_v, mean_f[:, 0])
        fit_green = scipy.stats.linregress(mean_v, mean_f[:, 1])
        fit_rat = scipy.stats.linregress(mean_v, mean_r)

        fits.append([fit_blue, fit_green, fit_rat])

        sens.append([fit_blue.slope, fit_green.slope, fit_rat.slope])

        if trial_string == example:
            end = 65
            ex_tc = dr_t[:, :-end]
            end_v = np.round((end / df_t.shape[-1]) * vm.shape[-1]).astype(int)
            vm_T = (df_t.shape[-1] / vm.shape[-1]) * T
            ex_vm = vm[:, :-end_v]
            ex_im = im[:, :-end_v]
            ii = idx

            st = tifffile.imread(
                [f for f in Path(data.directory).glob("./**/*.tif")][0]
            )
            image = np.mean(st[::2, ...], 0)
            _, roi = gf.read_roi_file(
                Path(
                    "/home/peter/data/Firefly/cancer/analysis/full",
                    "steps_analysis/rois",
                    f"{trial_string}.roi",
                ),
                im_dims=st.shape[-2:],
            )
            outline = np.logical_xor(roi, ndimage.binary_dilation(roi, iterations=2))

    mean_fs = np.array(mean_fs)
    mean_vs = np.array(mean_vs)
    mean_is = np.array(mean_is)
    mean_rs = np.array(mean_rs)

    sens = np.array(sens)

    sens = sens * 100**2

    disp = slice(10, 140, 1), slice(169, 222, 1)
    image = image[disp[0], disp[1]]
    outline = outline[disp[0], disp[1]]
    over = np.zeros(outline.shape + (4,), dtype=np.uint8)
    over[np.where(outline)] = (255, 0, 0, 255)

    length_um = 20
    length = np.round(length_um / 1.04).astype(int)
    over[5:8, -length - 5 : -5] = (255, 255, 255, 255)

    # plot an example cell
    fig = plt.figure(constrained_layout=True, figsize=[7, 4.8])
    gs = fig.add_gridspec(2, 3)

    figdata_name = "/home/peter/Dropbox/Papers/cancer/v2/figure_data/" + "fig_1_"
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(image, cmap="Greys_r")
    ax0.imshow(over)

    plt.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(np.arange(ex_vm.shape[-1]) * vm_T, ex_vm.T, linewidth=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Patch command\nvoltage (mV)")
    # ax1.set_yticks(np.arange(-1,4))
    pf.set_all_fontsize(ax1, 12)
    pf.set_thickaxes(ax1, 3)

    np.savetxt(figdata_name + f"_1D_x.txt", np.arange(ex_vm.shape[-1]) * vm_T)
    np.savetxt(figdata_name + f"_1D_y.txt", ex_vm.T)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(np.arange(ex_tc.shape[-1]) * T, (ex_tc.T - 1) * 100, linewidth=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"$\Delta R/R_0$ (%)")
    ax2.set_yticks(np.arange(-2, 7, 2))
    pf.set_all_fontsize(ax2, 12)
    pf.set_thickaxes(ax2, 3)

    np.savetxt(figdata_name + f"_1E_x.txt", np.arange(ex_tc.shape[-1]) * T)
    np.savetxt(figdata_name + f"_1E_y.txt", (ex_tc.T - 1) * 100)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(
        mean_vs[ii, :],
        (fits[ii][-1].slope * mean_vs[ii, :] + fits[ii][-1].intercept - 1) * 100,
        "k",
        linewidth=3,
    )
    for idx in range(mean_vs.shape[1]):
        ax3.plot(mean_vs[ii, idx], (mean_rs[ii, idx] - 1) * 100, ".", markersize=12)
    ax3.set_xlabel("Membrane Voltage (mV)")
    ax3.set_ylabel(r"$\Delta R/R_0$ (%)")
    ax3.set_yticks(np.arange(-2, 7, 2))
    ax3.set_xticks([-50, 0, 50])
    ax3.text(
        -60,
        2,
        f"{fits[ii][-1].slope*100**2:.1f} % per\n100 mV",
        fontdict={"fontsize": 12},
    )
    pf.set_all_fontsize(ax3, 12)
    pf.set_thickaxes(ax3, 3)

    np.savetxt(figdata_name + f"_1F_x.txt", mean_vs[ii, :])
    np.savetxt(figdata_name + f"_1F_y.txt", (mean_rs[ii, :] - 1) * 100)

    ax4 = fig.add_subplot(gs[1, 2])
    scale = 0.02
    sns.violinplot(y=sens[:, -1], saturation=0.5)
    # ax4.plot(np.random.normal(loc = 1,scale = scale,size = sens.shape[0]),sens[:,-1],'.k',markersize = 12)
    sns.swarmplot(y=sens[:, -1], ax=ax4, color="k", size=7)
    ax4.xaxis.set_visible(False)
    ax4.set_yticks(np.arange(2, 11, 2))
    ax4.set_ylabel("Ratiometric sensitivity\n(% per 100 mV)")
    pf.set_thickaxes(ax4, 3, remove=["top", "right", "bottom"])
    pf.set_all_fontsize(ax4, 12)

    np.savetxt(figdata_name + f"_1G_y.txt", sens[:, -1])

    fig.savefig(
        Path(figsave, f"patch_figure{filetype}"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

    # =============================================================================
    #
    #     # plot an example cell
    #     fig = plt.figure(constrained_layout=True, figsize=[7, 4.8])
    #     gs = fig.add_gridspec(2, 3)
    #
    #     ax0 = fig.add_subplot(gs[:, 0])
    #     ax0.imshow(image, cmap="Greys_r")
    #     ax0.imshow(over)
    #
    #     plt.axis("off")
    #
    #     ax1 = fig.add_subplot(gs[0, 1])
    #     ax1.plot(np.arange(ex_vm.shape[-1]) * vm_T, ex_vm.T, linewidth=2)
    #     ax1.set_xlabel("Time (s)")
    #     ax1.set_ylabel("Patch command\nvoltage (mV)")
    #     # ax1.set_yticks(np.arange(-1,4))
    #     pf.set_all_fontsize(ax1, 12)
    #     pf.set_thickaxes(ax1, 3)
    #
    #     ax2 = fig.add_subplot(gs[0, 2])
    #     ax2.plot(np.arange(ex_tc.shape[-1]) * T, (ex_tc.T - 1) * 100, linewidth=2)
    #     ax2.set_xlabel("Time (s)")
    #     ax2.set_ylabel(r"$\Delta R/R_0$ (%)")
    #     ax2.set_yticks(np.arange(-2, 7, 2))
    #     pf.set_all_fontsize(ax2, 12)
    #     pf.set_thickaxes(ax2, 3)
    #
    #     ax3 = fig.add_subplot(gs[1, 1])
    #     ax3.plot(
    #         mean_vs[ii, :],
    #         (fits[ii][-1].slope * mean_vs[ii, :] + fits[ii][-1].intercept - 1) * 100,
    #         "k",
    #         linewidth=3,
    #     )
    #     for idx in range(mean_vs.shape[1]):
    #         ax3.plot(mean_vs[ii, idx], (mean_rs[ii, idx] - 1) * 100, ".r", markersize=12)
    #     ax3.set_xlabel("Membrane Voltage (mV)")
    #     ax3.set_ylabel(r"$\Delta R/R_0$ (%)")
    #     ax3.set_yticks(np.arange(-2, 7, 2))
    #     ax3.set_xticks([-50, 0, 50])
    #     ax3.text(
    #         -60,
    #         2,
    #         f"{fits[ii][-1].slope*100**2:.1f} % per\n100 mV",
    #         fontdict={"fontsize": 12},
    #     )
    #     pf.set_all_fontsize(ax3, 12)
    #     pf.set_thickaxes(ax3, 3)
    #
    #     ax4 = fig.add_subplot(gs[1, 2])
    #     scale = 0.02
    #     sns.violinplot(y=sens[:, -1], saturation=0.5)
    #     # ax4.plot(np.random.normal(loc = 1,scale = scale,size = sens.shape[0]),sens[:,-1],'.k',markersize = 12)
    #     sns.swarmplot(y=sens[:, -1], ax=ax4, color="k", size=7)
    #
    #     ax4.xaxis.set_visible(False)
    #     ax4.set_yticks(np.arange(2, 11, 2))
    #     ax4.set_ylabel("Ratiometric sensitivity\n(% per 100 mV)")
    #     pf.set_thickaxes(ax4, 3, remove=["top", "right", "bottom"])
    #     pf.set_all_fontsize(ax4, 12)
    #
    #     fig.savefig(
    #         Path(figsave, f"patch_figure_red_dots{filetype}"),
    #         dpi=300,
    #         bbox_inches="tight",
    #         transparent=True,
    #     )
    #
    #     # plot an example cell
    #     fig = plt.figure(constrained_layout=True, figsize=[7, 4.8])
    #     gs = fig.add_gridspec(2, 3)
    #
    #     ax0 = fig.add_subplot(gs[:, 0])
    #     ax0.imshow(image, cmap="Greys_r")
    #     ax0.imshow(over)
    #
    #     plt.axis("off")
    #
    #     ax1 = fig.add_subplot(gs[0, 1])
    #     ax1.plot(np.arange(ex_vm.shape[-1]) * vm_T, ex_vm.T, linewidth=2)
    #     ax1.set_xlabel("Time (s)")
    #     ax1.set_ylabel("Patch command\nvoltage (mV)")
    #     # ax1.set_yticks(np.arange(-1,4))
    #     pf.set_all_fontsize(ax1, 12)
    #     pf.set_thickaxes(ax1, 3)
    #
    #     ax2 = fig.add_subplot(gs[0, 2])
    #     ax2.plot(np.arange(ex_tc.shape[-1]) * T, (ex_tc.T - 1) * 100, linewidth=2)
    #     ax2.set_xlabel("Time (s)")
    #     ax2.set_ylabel(r"$\Delta R/R_0$ (%)")
    #     ax2.set_yticks(np.arange(-2, 7, 2))
    #     pf.set_all_fontsize(ax2, 12)
    #     pf.set_thickaxes(ax2, 3)
    #
    #     ax3 = fig.add_subplot(gs[1, 1])
    #     ax3.plot(
    #         mean_vs[ii, :],
    #         (fits[ii][-1].slope * mean_vs[ii, :] + fits[ii][-1].intercept - 1) * 100,
    #         "k",
    #         linewidth=3,
    #     )
    #     for idx in range(mean_vs.shape[1]):
    #         ax3.plot(mean_vs[ii, idx], (mean_rs[ii, idx] - 1) * 100, ".r", markersize=12)
    #     ax3.set_xlabel("Membrane Voltage (mV)")
    #     ax3.set_ylabel(r"$\Delta R/R_0$ (%)")
    #     ax3.set_yticks(np.arange(-2, 7, 2))
    #     ax3.set_xticks([-50, 0, 50])
    #     ax3.text(
    #         -60,
    #         2,
    #         f"{fits[ii][-1].slope*100**2:.1f} % per\n100 mV",
    #         fontdict={"fontsize": 12},
    #     )
    #     pf.set_all_fontsize(ax3, 12)
    #     pf.set_thickaxes(ax3, 3)
    #
    #     ax4 = fig.add_subplot(gs[1, 2])
    #     scale = 0.02
    #     ax4.violinplot(sens[:, -1])
    #     ax4.plot(
    #         np.random.normal(loc=1, scale=scale, size=sens.shape[0]),
    #         sens[:, -1],
    #         ".k",
    #         markersize=12,
    #     )
    #     # sns.swarmplot(y=sens[:,-1],ax = ax4,color = 'k',size = 7)
    #
    #     ax4.xaxis.set_visible(False)
    #     ax4.set_yticks(np.arange(2, 11, 2))
    #     ax4.set_ylabel("Ratiometric sensitivity\n(% per 100 mV)")
    #     pf.set_thickaxes(ax4, 3, remove=["top", "right", "bottom"])
    #     pf.set_all_fontsize(ax4, 12)
    #
    #     fig.savefig(
    #         Path(figsave, f"patch_figure_old_violin{filetype}"),
    #         dpi=300,
    #         bbox_inches="tight",
    #         transparent=True,
    #     )
    # =============================================================================

    datafile = Path(figsave, "info.txt")

    with open(datafile, "w") as f:
        f.write(f"Scale bar image is {length_um} um\n")
        f.write(
            f"Ratio sensitivity mean = {sens[:,-1].mean()} sem. {sens[:,-1].std()/np.sqrt(sens.shape[0])}\n"
        )


if __name__ == "__main__":
    figure_dir = Path("/home/peter/Dropbox/Papers/cancer/v2/")
    make_figures(figure_dir)
