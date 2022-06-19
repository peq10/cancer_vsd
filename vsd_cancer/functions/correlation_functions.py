#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 14:28:50 2021

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import astropy.stats as ass
import warnings

print("Should I be taking the absolute value??")


def get_trains(spike_trains):
    """
    This interprets how I saved the data - returns a list of spike trains, cell positions and cell ids
    """

    trains = []
    pos = []
    ids = []

    for cell in spike_trains.keys():
        trains.append(spike_trains[cell][0])
        pos.append(spike_trains[cell][1])
        ids.append(cell)

    return trains, np.array(pos), ids


def bin_times(trains, binsize):
    """
    Bins all the spikes for the whole time
    """
    min_time = 400 * 0.2
    max_time = np.max([np.max(x) for x in trains])
    time_bins = np.arange(min_time, max_time + binsize, binsize)
    binned_spikes = np.array([np.histogram(x, bins=time_bins)[0] for x in trains])
    binned_spikes[binned_spikes > 1] = 1

    if np.any(np.all(binned_spikes == 1, axis=1)):
        binned_spikes = np.pad(
            binned_spikes, ((0, 0), (1, 0))
        )  # this removes an error where the std is 0 so get nan

    return binned_spikes


def calculate_pairwise_corrs(binned_spikes):

    n = binned_spikes.shape[0]
    result = np.zeros(int((n * (n - 1) / 2)))

    for idx0, (idx1, idx2) in enumerate(itertools.combinations(range(n), 2)):
        result[idx0] = np.abs(
            np.corrcoef(binned_spikes[idx1, :], binned_spikes[idx2, :])[0, 1]
        )
        if result[idx0] > 0.98 and False:
            plt.show()
            plt.plot(binned_spikes[idx1, :])
            plt.plot(binned_spikes[idx2, :] + 1)
            plt.show()

    return result


def get_all_pairwise(all_trains, binsize):

    all_res = []

    for x in all_trains:
        if len(x) == 1:
            continue

        trains, _, _ = get_trains(x)
        binned_spikes = bin_times(trains, binsize)

        all_res += list(calculate_pairwise_corrs(binned_spikes))

    return all_res


def get_null_hypoth(all_trains, binsize, repeats=10):
    rng = np.random.default_rng()
    all_res = [[] for x in range(repeats)]

    for x in all_trains:
        if len(x) == 1:
            continue

        trains, _, _ = get_trains(x)
        binned_spikes = bin_times(trains, binsize)

        for x in range(repeats):
            [rng.shuffle(x) for x in binned_spikes]

            all_res[x] += list(calculate_pairwise_corrs(binned_spikes))

    return all_res


def plot_raster(trains, binsize=1):
    min_time = np.min([np.min(x) for x in trains])
    for idx, t in enumerate(trains):
        plt.plot((t - min_time) / binsize, np.ones_like(t) + idx, ".")


def resample_observations(all_trains, binsize, bootnum=10**2):
    """
    Resamopling from all pairs to understand the 95% CIs
    """

    binned = []
    pairs = []
    for idx, x in enumerate(all_trains):
        if len(x) == 1:
            binned.append([])
            continue

        trains, _, _ = get_trains(x)
        binned_spikes = bin_times(trains, binsize)
        binned.append(binned_spikes)

        # getting a list of all possible pairs
        n = binned_spikes.shape[0]
        pairs += [(idx,) + pair for pair in itertools.combinations(range(n), 2)]

    # now resample the pairs and calculate the corr coeff
    resamples = ass.bootstrap(np.arange(len(pairs)), bootnum=bootnum).astype(int)
    resamp_means = np.zeros(bootnum)

    for idxx, res in enumerate(resamples):
        # get corr coeffs for resampled pairs
        ccs = np.zeros(resamples.shape[1])
        for idx, pairnum in enumerate(res):
            ids = pairs[pairnum]
            ccs[idx] = np.abs(
                np.corrcoef(binned[ids[0]][ids[1], :], binned[ids[0]][ids[2], :])[0, 1]
            )
            if np.isnan(ccs[idx]):
                raise ValueError("hmmm")

        resamp_means[idxx] = np.mean(ccs)
        if np.isnan(resamp_means[idxx]):
            raise ValueError("hmmm")

    return resamp_means


def resample_observations_both(all_trains, binsize, bootnum=10**2):
    """
    Resamopling from all pairs to understand the 95% CIs
    """
    rng = np.random.default_rng()
    binned = []
    pairs = []
    for idx, x in enumerate(all_trains):
        if len(x) == 1:
            binned.append([])
            continue

        trains, _, _ = get_trains(x)
        binned_spikes = bin_times(trains, binsize)
        binned.append(binned_spikes)

        # getting a list of all possible pairs
        n = binned_spikes.shape[0]
        pairs += [(idx,) + pair for pair in itertools.combinations(range(n), 2)]

    # now resample the pairs and calculate the corr coeff
    resamples = ass.bootstrap(np.arange(len(pairs)), bootnum=bootnum).astype(int)
    resamp_means = np.zeros(bootnum)
    resamp_means_null = np.zeros(bootnum)

    for idxx, res in enumerate(resamples):
        # get corr coeffs for resampled pairs
        ccs = np.zeros(resamples.shape[1])
        ccs_null = np.zeros(resamples.shape[1])
        for idx, pairnum in enumerate(res):
            ids = pairs[pairnum]
            ccs[idx] = np.abs(
                np.corrcoef(binned[ids[0]][ids[1], :], binned[ids[0]][ids[2], :])[0, 1]
            )

            b1 = rng.permutation(binned[ids[0]][ids[1], :])
            b2 = rng.permutation(binned[ids[0]][ids[2], :])

            ccs_null[idx] = np.abs(np.corrcoef(b1, b2)[0, 1])

        resamp_means[idxx] = np.mean(ccs)
        resamp_means_null[idxx] = np.mean(ccs_null)

    return resamp_means, resamp_means_null


def get_ratio_corr_CIs(
    all_trains, binsize, level, bootnum=10**3, shuffle=False, plot=True
):
    resamp, resamp_null = resample_observations_both(all_trains, binsize)

    ratio = resamp / resamp_null
    CI = np.percentile(ratio, level / 2), np.percentile(ratio, 100 - level / 2)

    return CI, resamp, resamp_null


def get_corr_CIs(all_trains, binsize, level, bootnum=10**3, shuffle=False, plot=True):
    resamplings = resample_observations(all_trains, binsize, bootnum=bootnum)

    # todo - check about percentile vs other bootstrap
    CI = (
        np.percentile(resamplings, level / 2),
        np.percentile(resamplings, 100 - level / 2),
    )

    if plot:
        real = np.mean(get_all_pairwise(all_trains, binsize))
        fig, ax = plt.subplots()
        h = ax.hist(resamplings, bins=25)
        ax.plot([real, real], np.array([0, 1]) * h[0].max())

    return CI, resamplings


def calculate_p_value(all_trains, binsize, bootnum=10**2):

    nulls = get_null_hypoth(all_trains, binsize, repeats=bootnum)
    null_dist = np.array([np.mean(x) for x in nulls])

    ours = get_all_pairwise(all_trains, binsize)
    ours_mean = np.mean(ours)

    if True:
        fig, ax = plt.subplots()
        h = ax.hist(null_dist, bins=25)
        ax.plot([ours_mean, ours_mean], np.array([0, 1]) * h[0].max())

    # one sided hypothesis test for our correlation being greater than normal
    p_value = len(null_dist[null_dist > ours_mean]) / len(null_dist)

    if len(null_dist[null_dist > ours_mean]) == 0:
        warnings.warn("Not enough resamples to resolve p this small")
        p_value = 1 / bootnum  # we can only resolve to be up to this val

    return p_value, null_dist


def get_speed_distribution(all_trains, binsize, shuffle=False):
    rng = np.random.default_rng()

    px_sz = 1.04

    def dist(pos1, pos2):
        return np.sqrt(np.sum((pos1 - pos2) ** 2))

    binned = []
    pairs = []

    for idx, x in enumerate(all_trains):
        if len(x) == 1:
            binned.append([])
            continue

        trains, pos, _ = get_trains(x)
        binned_spikes = bin_times(trains, binsize)
        binned.append(binned_spikes)

        # getting a list of all possible pairs
        n = binned_spikes.shape[0]
        pairs += [
            (idx,) + pair + (dist(pos[pair[0]], pos[pair[1]]),)
            for pair in itertools.combinations(range(n), 2)
        ]

    corrs = []
    times = []
    inv_speeds = []
    dists = []
    data = []

    for i, p in enumerate(pairs):
        b1 = binned[p[0]][p[1], :]
        b2 = binned[p[0]][p[2], :]

        if shuffle:
            rng.shuffle(b1)
            rng.shuffle(b2)

        corr = np.correlate(b1, b2, mode="full")

        corrs.append(corr)

        dists.append(p[3] * px_sz)

        # need to convert from time to speed
        time = np.arange(0, (2 * len(b1) - 1) * binsize, binsize) - len(b1) * binsize
        inv_speed = time / p[3]

        times.append(time)
        inv_speeds.append(inv_speed)

        # now bin the data

        wh = np.argwhere(corr)[:, 0]

        speeds = inv_speed[wh]
        vals = corr[wh]
        tt = time[wh]

        data += list(np.array((speeds, vals, tt)).T)

    data = np.array(data)

    return data, (corrs, dists, times)
