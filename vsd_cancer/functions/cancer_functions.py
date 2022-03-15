#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:12:06 2020

@author: peter
"""
import numpy as np
from pathlib import Path
import shutil
import json

import tifffile
import quantities as pq
import scipy.interpolate as interp
import scipy.ndimage as ndimage
import scipy.signal as signal
import pandas as pd
import datetime
import pdb

import re

import f.general_functions as gf
import f.ephys_functions as ef


def get_events_exclude_surround_events(
    tc,
    std,
    surround_tc,
    surround_std,
    z_score=3,
    surround_z=7,
    exclude_first=0,
    max_overlap=0.75,
    excluded_circle=None,
    excluded_dead=None,
):

    ev = detect_events(tc, std, z_score=z_score, exclude_first=exclude_first)

    surrounds_ev = detect_events(
        tc, std, z_score=surround_z, exclude_first=exclude_first
    )

    excluded_dict = {}
    dict_drop = []
    for key in ev.keys():
        if type(key) == str:
            continue

        if key not in surrounds_ev.keys():
            continue

        sur_e = surrounds_ev[key].T
        e = ev[key].T
        # if a detected surround event overlaps for more than max_overlap, then remove

        # detects any overlaps
        overlapping = np.logical_and(
            e[:, 0, None] < sur_e[None, :, 1], e[:, 1, None] >= sur_e[None, :, 0]
        )

        if not np.any(overlapping):
            continue

        drop = []
        wh = np.where(overlapping)
        # now detect size of overlap and delete if proportionally greater than max overlap
        for idx in range(len(wh[0])):
            overlap = min(e[wh[0][idx], 1], sur_e[wh[1][idx], 1]) - max(
                e[wh[0][idx], 0], sur_e[wh[1][idx], 0]
            )
            if overlap > max_overlap * (e[wh[0][idx], 1] - e[wh[0][idx], 0]):
                drop.append(wh[0][idx])

        # pdb.set_trace()

        exc_e = np.array([x for ii, x in enumerate(e) if ii in drop])
        keep_e = np.array([x for ii, x in enumerate(e) if ii not in drop])

        excluded_dict[key] = exc_e.T

        if len(keep_e) > 0:
            ev[key] = keep_e.T
        else:
            dict_drop.append(key)

    # delete empty fields

    for key in dict_drop:
        del ev[key]

    # exclude ROIs on edge of illumination
    if excluded_circle is not None:
        circle_dict = {}
        for idx in excluded_circle:
            if idx in ev.keys():
                circle_dict[idx] = ev[idx]
                del ev[idx]

        ev["excluded_circle_events"] = circle_dict

    # exclude ROIs on edge of illumination
    if excluded_dead is not None:
        dead_dict = {}
        if len(excluded_dead) > 0:
            for idx in excluded_dead:
                if idx in ev.keys():
                    dead_dict[idx] = ev[idx]
                    del ev[idx]

        else:
            pass
        ev["excluded_dead_events"] = dead_dict

    # include the surround data
    ev["surround_events"] = surrounds_ev
    ev["excluded_events"] = excluded_dict

    return ev


def get_events_exclude_simultaneous_events(
    tc,
    std,
    z_score=3,
    exclude_first=0,
    max_events=5,
    overlap=0.75,
    excluded_circle=None,
    excluded_dead=None,
):

    ev, excluded_dict = detect_events_remove_simultaneous(
        tc,
        std,
        z_score=z_score,
        exclude_first=exclude_first,
        max_overlap=overlap,
        max_events=max_events,
    )

    # exclude ROIs on edge of illumination
    if excluded_circle is not None:
        circle_dict = {}
        for idx in excluded_circle:
            if idx in ev.keys():
                circle_dict[idx] = ev[idx]
                del ev[idx]

        ev["excluded_circle_events"] = circle_dict

    # exclude ROIs on edge of illumination
    if excluded_dead is not None:
        dead_dict = {}
        if len(excluded_dead) > 0:
            for idx in excluded_dead:
                if idx in ev.keys():
                    dead_dict[idx] = ev[idx]
                    del ev[idx]

        else:
            pass
        ev["excluded_dead_events"] = dead_dict

    ev["excluded_events"] = excluded_dict
    ev["surround_events"] = excluded_dict
    print("Check this - surrounds and exclude the same")

    return ev


def detect_events_remove_simultaneous(
    tc, std, z_score=3, exclude_first=0, max_events=5, max_overlap=0.5
):

    tc_filt = ndimage.gaussian_filter(tc, (0, 3))
    std_filt = ndimage.gaussian_filter(std, (0, 3))

    tc_filt[:, :exclude_first] = 1

    events = np.abs(tc_filt - 1) > z_score * std_filt

    # Use closing to join split events and remove small events
    struc = np.zeros((3, 5))
    struc[1, :] = 1
    events = ndimage.binary_opening(events, structure=struc, iterations=2)
    events = ndimage.binary_closing(events, structure=struc, iterations=2)

    # now count simultaneous events and remove those where they are
    num_events = np.sum(events, 0)
    excluded_events = num_events > max_events
    excluded_time = np.where(excluded_events)[0]

    wh = np.where(events)
    idxs, locs = np.unique(wh[0], return_index=True)
    locs = np.append(locs, len(wh[0]))

    excluded_result = {}
    result = {}
    for i, idx in enumerate(idxs):
        llocs = wh[1][locs[i] : locs[i + 1]]
        split_locs = np.array(recursive_split_locs(llocs))
        # check if they have both positive and negative going - messes with integration later
        t = tc_filt[idx, :]
        corr_locs = correct_event_signs(t, split_locs)

        overlap = np.sum(np.isin(llocs, excluded_time).astype(int)) / len(llocs)
        if overlap > max_overlap:
            excluded_result[idx] = corr_locs.T
        else:
            result[idx] = corr_locs.T

    result["tc_filt"] = tc_filt
    result["tc"] = tc
    return result, excluded_result


def get_surround_masks(masks, surround_rad=20, dilate=True):
    def get_bounding_circle_radius(masks):
        rows, cols = np.any(masks, axis=-1), np.any(masks, axis=-2)
        rs = np.apply_along_axis(first_last, -1, rows)
        cs = np.apply_along_axis(first_last, -1, cols)

        centers = np.array(
            [rs[:, 0] + (rs[:, 1] - rs[:, 0]) / 2, cs[:, 0] + (cs[:, 1] - cs[:, 0]) / 2]
        ).T
        # bounding radius is the hypotenuse /2
        radii = np.sqrt((cs[:, 0] - cs[:, 0]) ** 2 + (rs[:, 1] - rs[:, 0]) ** 2) / 2
        return radii, centers

    def first_last(arr_1d):
        return np.where(arr_1d)[0][[0, -1]]

    # avoid border effects/bleedthrough by dilating existing rois
    structure = np.ones((3, 3, 3))
    structure[0::2, ...] = 0
    dilated_masks = ndimage.binary_dilation(masks, structure=structure, iterations=4)

    roi_rads, centers = get_bounding_circle_radius(dilated_masks)
    x, y = np.indices(masks.shape[-2:])
    rs = np.sqrt(
        (x[None, ...] - centers[:, 0, None, None]) ** 2
        + (y[None, ...] - centers[:, 1, None, None]) ** 2
    )

    surround_roi = np.logical_xor(
        dilated_masks, rs < roi_rads[:, None, None] + surround_rad
    )
    return surround_roi


def get_surround_masks_cellfree(masks, surround_rad=50, dilate=True):

    all_masks = np.any(masks, axis=0)

    # avoid border effects/bleedthrough by dilating existing rois
    structure = np.ones((3, 3, 3))
    structure[0::2, ...] = 0
    dilated_masks = ndimage.binary_dilation(masks, structure=structure, iterations=4)

    centers = np.array([ndimage.center_of_mass(m) for m in dilated_masks])
    x, y = np.indices(masks.shape[-2:])
    rs = np.sqrt(
        (x[None, ...] - centers[:, 0, None, None]) ** 2
        + (y[None, ...] - centers[:, 1, None, None]) ** 2
    )

    surround_roi = np.logical_and(~all_masks, rs < surround_rad)

    # see if the area is too small
    areas = np.sum(surround_roi, axis=(-2, -1))

    # check nowhere too small
    small = areas < 2000
    if np.any(small):
        for new_rs in range(surround_rad, 2 * surround_rad, 10):
            small = areas < 2000
            surround_roi[small] = np.logical_and(~all_masks, rs[small, ...] < new_rs)
            if not np.any(small):
                break

    small = areas < 2000
    # revert back to normal behaviour - just take an area around and dont care about cells
    if np.any(small):
        surround_roi[small] = np.logical_and(masks[small], rs[small, ...] < new_rs)

    return surround_roi


def get_observation_length(event_dict):
    tc = event_dict["tc_filt"]
    exclude_dict = event_dict["surround_events"]
    length = tc.shape[1]
    lengths = []
    # count as non-observed any time during a surround event
    for i in range(tc.shape[0]):
        if i in exclude_dict.keys():
            lengths.append(
                length - np.sum(exclude_dict[i].T[:, 1] - exclude_dict[i].T[:, 0])
            )
        else:
            lengths.append(length)

    return np.array(lengths)


def apply_exclusion(exclude_dict, tc):
    excluded_tc = np.copy(tc)

    for roi in exclude_dict.keys():
        for i in range(exclude_dict[roi].shape[-1]):
            ids = exclude_dict[roi][:, i]
            excluded_tc[roi, ids[0] : ids[1]] = 1

    return excluded_tc


def soft_threshold(arr, thresh, to=1):
    # Thresholds towards to value
    res = np.copy(arr)
    wh = np.where(np.abs(arr - to) < thresh)
    n_wh = np.where(np.abs(arr - to) >= thresh)
    sgn = np.sign(arr - to)
    res[wh] = to
    res[n_wh] -= sgn[n_wh] * thresh

    return res


def split_event(t, ids):
    # splits a zero-(actually 1) crossing event into multiple non-zero crossing events recursively
    # removes one point
    if not np.logical_and(
        np.any(t[ids[0] : ids[1]] - 1 > 0), np.any(t[ids[0] : ids[1]] - 1 < 0)
    ):
        return [tuple(ids)]
    else:
        zer_loc = np.argmin(np.abs(t[ids[0] : ids[1]] - 1)) + ids[0]
        return split_event(t, (ids[0], zer_loc)) + split_event(t, (zer_loc + 1, ids[1]))


def correct_event_signs(t, llocs):
    corr_locs = []
    for id_idx, ids in enumerate(llocs):
        if np.logical_and(
            np.any(t[ids[0] : ids[1]] - 1 > 0), np.any(t[ids[0] : ids[1]] - 1 < 0)
        ):
            split_ids = split_event(t, ids)
            corr_locs.extend(split_ids)
        else:
            corr_locs.append(ids)

    corr_locs = np.array(corr_locs)

    # if we have split into a zero size (due to boundary issue in split events), remove
    if np.any((corr_locs[:, 1] - corr_locs[:, 0]) < 1):
        corr_locs = corr_locs[(corr_locs[:, 1] - corr_locs[:, 0]) > 0]
    return corr_locs


def recursive_split_locs(seq):
    # splits a sequence into n adjacent sequences
    diff = np.diff(seq)
    if not np.any(diff != 1):
        return [(seq[0], seq[-1])]
    else:
        wh = np.where(diff != 1)[0][0] + 1
        return recursive_split_locs(seq[:wh]) + recursive_split_locs(seq[wh:])


def detect_events(tc, std, z_score=3, exclude_first=0):

    tc_filt = ndimage.gaussian_filter(tc, (0, 3))
    std_filt = ndimage.gaussian_filter(std, (0, 3))

    tc_filt[:, :exclude_first] = 1

    events = np.abs(tc_filt - 1) > z_score * std_filt

    # Use closing to join split events and remove small events
    struc = np.zeros((3, 5))
    struc[1, :] = 1
    events = ndimage.binary_opening(events, structure=struc, iterations=2)
    events = ndimage.binary_closing(events, structure=struc, iterations=2)

    wh = np.where(events)
    idxs, locs = np.unique(wh[0], return_index=True)
    locs = np.append(locs, len(wh[0]))

    result = {}
    for i, idx in enumerate(idxs):
        llocs = wh[1][locs[i] : locs[i + 1]]
        split_locs = np.array(recursive_split_locs(llocs))
        # check if they have both positive and negative going - messes with integration later
        t = tc_filt[idx, :]
        corr_locs = correct_event_signs(t, split_locs)
        result[idx] = corr_locs.T

    result["tc_filt"] = tc_filt
    result["tc"] = tc
    return result


def get_event_properties(event_dict, use_filt=True):
    if use_filt:
        t = event_dict["tc"]
    else:
        t = event_dict["tc_filt"]

    result_dict = {}

    for idx in event_dict.keys():
        if type(idx) == str:
            continue
        event_properties = []
        for locs in event_dict[idx].T:
            if np.logical_and(
                np.any(t[idx, locs[0] : locs[1]] - 1 > 0),
                np.any(t[idx, locs[0] : locs[1]] - 1 < 0),
            ):
                print(idx, locs)
                raise ValueError("This shouldnt happen")

            event_length = locs[1] - locs[0]
            event_amplitude = (
                t[idx, np.argmax(np.abs(t[idx, locs[0] : locs[1]] - 1)) + locs[0]] - 1
            )

            event_integrated = np.sum(t[idx, locs[0] : locs[1]] - 1)
            event_properties.append([event_length, event_amplitude, event_integrated])

        if len(np.array(event_properties)) == 0:
            pdb.set_trace()
        result_dict[idx] = np.array(event_properties)

    event_dict["event_props"] = result_dict

    return event_dict


def lab2masks(seg):
    masks = []
    for i in range(1, seg.max() + 1):
        masks.append((seg == i).astype(int))
    return np.array(masks)


def t_course_from_roi(nd_stack, roi):
    if len(roi.shape) != 2:
        raise NotImplementedError("Only works for 2d ROIs")
    wh = np.where(roi)
    return np.mean(nd_stack[..., wh[0], wh[1]], -1)


def median_t_course_from_roi(nd_stack, roi):
    if len(roi.shape) != 2:
        raise NotImplementedError("Only works for 2d ROIs")
    wh = np.where(roi)
    return np.median(nd_stack[..., wh[0], wh[1]], -1)


def std_t_course_from_roi(nd_stack, roi, standard_err):
    """
    Gets the standard deviation of the pixels in the roi at each time point

    Parameters
    ----------
    nd_stack : TYPE
        DESCRIPTION.
    roi : TYPE
        DESCRIPTION.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if len(roi.shape) != 2:
        raise NotImplementedError("Only works for 2d ROIs")
    wh = np.where(roi)

    if standard_err:
        fac = 1 / np.sqrt(np.sum(roi))
    else:
        fac = 1

    return fac * np.std(nd_stack[..., wh[0], wh[1]], -1)


def load_tif_metadata(fname):
    fname = Path(fname)
    metadata_file = Path(fname.parent, Path(fname.stem).stem + "_metadata.txt")
    if "rds" in str(Path.home()):
        to_file = metadata_file
    else:
        to_file = Path("/tmp/tmp_metadata.txt")
        shutil.copy(
            metadata_file, to_file
        )  # this is to deal with a wierd bug due to NTFS filesystem?

    with open(to_file, "r") as f:
        metadict = json.load(f)
    return metadict


def parse_time(metadata_time):
    date = metadata_time.split(" ")[0].split("-")
    time = metadata_time.split(" ")[1].split(":")
    return int("".join(date)), time


def lin_time(time):
    return float(time[0]) * 60 ** 2 + float(time[1]) * 60 + float(time[2])


def get_stack_offset(fname, ephys_start):
    date, time = parse_time(load_tif_metadata(fname)["Summary"]["StartTime"])

    if int(date) != int(ephys_start[0]):
        raise ValueError("Date mismatch!")

    ttime = [
        str(ephys_start[1])[:2],
        str(ephys_start[1])[2:4],
        str(ephys_start[1])[4:6],
    ]
    offset = lin_time(time) - lin_time(ttime)
    if offset < 0:
        raise ValueError("Time mismatch!")

    return offset


def slice_cam(cam_frames, n_frames, n_repeats, T):
    starts = np.where(np.concatenate(([1], np.diff(cam_frames) > 2 * T)))[0]
    # remove any consecutive and take last

    starts = starts[np.concatenate((~(np.diff(starts) == 1), [True]))]
    sliced_frames = np.zeros((n_repeats, n_frames))
    for idx in range(n_repeats):
        st = starts[idx]
        sliced_frames[idx, ...] = cam_frames[st : st + n_frames]

    if np.any(np.diff(sliced_frames, axis=-1) > 2 * T):
        raise ValueError("Frames not sliced properly")
    return sliced_frames


def slice_ephys(analog_signal, single_cam):
    idx0 = ef.time_to_idx(analog_signal, single_cam[0] * pq.s)
    idx1 = ef.time_to_idx(analog_signal, single_cam[-1] * pq.s)
    return analog_signal[idx0:idx1]


def slice_all_ephys(analog_signal, sliced_cam):
    all_ephys = []
    sh = len(analog_signal)
    for ca in sliced_cam:
        ep = slice_ephys(analog_signal, ca)
        if len(ep) < sh:
            sh = len(ep)
        all_ephys.append(ep)
    return np.array([np.squeeze(all_ephys[i][:sh]) for i in range(len(all_ephys))])


def get_steps_image_ephys(im_dir, ephys_fname):
    ephys_dict = ef.load_ephys_parse(
        ephys_fname, analog_names=["vcVm", "vcIm"], event_names=["CamDown"]
    )

    files = [f for f in Path(im_dir).glob("./**/*.tif")]
    offsets = np.array([get_stack_offset(f, ephys_dict["ephys_start"]) for f in files])

    offsets, files = gf.sort_zipped_lists([offsets, files])

    for idx, f in enumerate(files):
        stack = tifffile.imread(f)

        if idx == 0:
            stacks = np.zeros(((len(files),) + stack.shape), dtype=np.uint16)

        stacks[idx, ...] = stack

    metadata = load_tif_metadata(files[0])
    T = float(metadata["FrameKey-0-0-0"]["HamamatsuHam_DCAM-Exposure"]) * 10 ** -3

    cam = ephys_dict["cam"]
    cam = cam[
        np.logical_and(
            cam > offsets[0] - 10, cam < offsets[-1] + stacks.shape[1] * T + 10
        )
    ]

    sliced_cam = slice_cam(cam, stacks.shape[1], stacks.shape[0], T)

    ephys_dict["sliced_cam"] = sliced_cam
    ephys_dict["cam"] = cam

    if np.any(np.diff(sliced_cam[:, 0] - offsets) > stacks.shape[1] * T):
        raise ValueError("Problemo!")

    # now slice the ephys from the cam
    for key in ["vcVm", "ccVm", "ccIm", "vcIm"]:
        if key not in ephys_dict.keys():
            continue
        ephys_dict[key + "_sliced"] = slice_all_ephys(ephys_dict[key], sliced_cam)
        idx0 = ef.time_to_idx(ephys_dict[key], offsets[0] - 10)
        idx1 = ef.time_to_idx(ephys_dict[key], offsets[-1] + 10)
        ephys_dict[key] = ephys_dict[key][idx0:idx1]

    return ephys_dict, stacks


def process_ratio_stacks(stacks):
    """
    assumes dims = (....,t,y,x)
    """
    sh = stacks.shape
    stacks = stacks.reshape((-1,) + sh[-3:])
    res = np.zeros((stacks.shape[0], 2) + sh[-3:]).astype(float)
    for idx, st in enumerate(stacks):
        res[idx, ...] = interpolate_stack(st)

    return res.reshape(sh[:-3] + (2,) + sh[-3:])


def interpolate_stack(ratio_stack, framelim=1000):
    nits = int(np.ceil(ratio_stack.shape[0] / framelim))

    full_res = np.zeros((2,) + ratio_stack.shape)
    for it in range(nits):
        stack = ratio_stack[it * framelim : (it + 1) * framelim, ...]
        result = np.zeros((2,) + stack.shape)
        y, x = (
            np.arange(stack.shape[1], dtype=int),
            np.arange(stack.shape[2], dtype=int),
        )
        z = [
            np.arange(0, stack.shape[0], 2, dtype=int),
            np.arange(1, stack.shape[0], 2, dtype=int),
        ]
        for i in range(2):
            j = np.mod(i + 1, 2)
            result[i, i::2, ...] = stack[i::2, ...]
            interped = interp.RegularGridInterpolator(
                (z[i], y, x), stack[i::2, ...], bounds_error=False, fill_value=None
            )
            pts = np.indices(stack.shape, dtype=int)[:, j::2, ...].reshape((3, -1))
            result[i, j::2, ...] = interped(pts.T).reshape(stack[1::2, ...].shape)

        full_res[:, it * framelim : it * framelim + result.shape[1], ...] = result

    return full_res


def get_LED_powers(LED, cam, T_approx, cam_edge="falling"):
    # assumes LED and cam contain only sliced vals, cam is camDown
    if cam_edge != "falling":
        raise NotImplementedError("Only implemented for cam falling edge")

    # do a rough pass then a second to get LED real value
    ids = ef.time_to_idx(
        LED,
        [
            cam[1] + T_approx,
            cam[1] + 3 * T_approx,
            cam[0] - T_approx,
            cam[0],
            cam[1] - T_approx,
            cam[1],
        ],
    )
    zer = LED[ids[0] : ids[1]].magnitude.mean()
    l1 = LED[ids[2] : ids[3]].magnitude.mean()
    l2 = LED[ids[4] : ids[5]].magnitude.mean()
    thr = 0.5 * (zer + min(l1, l2)) + zer

    LED_thr = LED > thr

    ##get actual T
    T = (np.sum(LED_thr.astype(int)) / len(cam)) / LED.sampling_rate.magnitude

    if np.abs(T - T_approx) > T_approx / 2:
        print(T)
        print(T_approx)
        print("Problems?")

    # now get accurate values
    ids1 = np.array(
        [
            ef.time_to_idx(LED, cam[::2] - 3 * T / 4),
            ef.time_to_idx(LED, cam[::2] - T / 4),
        ]
    ).T
    led1 = np.mean([LED[x[0] : x[1]].magnitude.mean() for x in ids1])

    ids2 = np.array(
        [
            ef.time_to_idx(LED, cam[1::2] - 3 * T / 4),
            ef.time_to_idx(LED, cam[1::2] - T / 4),
        ]
    ).T
    led2 = np.mean([LED[x[0] : x[1]].magnitude.mean() for x in ids2])

    ids3 = np.array(
        [ef.time_to_idx(LED, cam[1:-1:2] + T), ef.time_to_idx(LED, cam[2::2] - 5 * T)]
    ).T
    zer = np.mean([LED[x[0] : x[1]].magnitude.mean() for x in ids3])

    led1 -= zer
    led2 -= zer

    return led1, led2


def cam_check(cam, cam_id, times, e_start, fs):
    if cam_id + len(times) > len(cam):
        print("length issue")
        return False

    if len(times) % 2 == 1:
        times = times[:-1]

    cam_seg = cam[cam_id : cam_id + len(times)]
    IFI = np.array([np.diff(cam_seg[::2]), np.diff(cam_seg[1::2])])

    # check frame rate consistent
    if np.any(np.abs(IFI - 1 / fs) > (1 / fs) / 100):
        print("IFI issue")
        return False

    # compare our segment with if we are off by one each direction - are we at a minimum?
    if cam_id + len(times) == len(cam):
        if cam_id == 0:  # exactly 10000 frames
            return True
        v = [-1, 0]
    elif cam_id == 0:
        v = [0, 1]
    else:
        v = [-1, 0, 1]

    var = [
        np.std(cam[cam_id + x : cam_id + x + len(times)] + e_start - times) for x in v
    ]
    if var[1] != min(var) and cam_id != 0:
        print("Bad times?")
        return False
    elif var[0] != min(var) and cam_id == 0:
        print("Bad times?")
        return False

    return True


def save_result_hdf(hdf_file, result_dict, group=None):
    f = hd5py.File(hdf_file, "a")

    if group is not None:
        group = f'{group}/{to_trial_string(result_dict["tif_file"])}'
    else:
        group = f'{to_trial_string(result_dict["tif_file"])}'

    grp = f.create_group(group)

    for key in result_dict.keys():
        t = type(result_dict[key])
        if t == "neo.core.analogsignal.AnalogSignal":
            print(0)
        elif t == "numpy.ndarray":
            print(1)
        else:
            raise NotImplementedError("Implement this")


def get_all_frame_times(metadict):
    frames = []
    times = []
    for k in metadict.keys():
        if k == "Summary":
            continue

        frame = int(k.split("-")[1])
        frames.append(frame)
        time = (
            metadict[k]["UserData"]["TimeReceivedByCore"]["scalar"]
            .split(" ")[1]
            .split(":")
        )
        time = float(time[0]) * 60 ** 2 + float(time[1]) * 60 + float(time[2])
        times.append(time)

    frames, times = gf.sort_zipped_lists([frames, times])

    return np.array(frames), np.array(times)


def load_and_slice_long_ratio(
    stack_fname, ephys_fname, T_approx=3 * 10 ** -3, fs=5, washin=False, nofilt=False
):
    stack = tifffile.imread(stack_fname)

    n_frames = len(stack)

    if Path(ephys_fname).is_file():
        ephys_dict = ef.load_ephys_parse(
            ephys_fname, analog_names=["LED", "vcVm"], event_names=["CamDown"]
        )

        e_start = [
            float(str(ephys_dict["ephys_start"][1])[i * 2 : (i + 1) * 2])
            for i in range(3)
        ]
        e_start[-1] += (float(ephys_dict["ephys_start"][2]) / 10) / 1000
        e_start = lin_time(e_start)

        meta = load_tif_metadata(stack_fname)
        frames, times = get_all_frame_times(meta)

        cam = ephys_dict["CamDown_times"]
        cam_id = np.argmin(np.abs(cam + e_start - times[0]))

        if not cam_check(cam, cam_id, times, e_start, fs):
            if cam_check(cam, cam_id - 1, times, e_start, fs):
                print("sub 1")
                cam_id -= 1
            elif cam_check(cam, cam_id + 1, times, e_start, fs):
                print("plus 1")
                cam_id += 1
            elif cam_check(cam, cam_id - 2, times, e_start, fs):
                print("sub 2")
                cam_id -= 2
            else:

                raise ValueError("possible bad segment")

        cam = cam[cam_id : cam_id + n_frames]

        # extract LED powers (use slightly longer segment)
        idx1, idx2 = ef.time_to_idx(
            ephys_dict["LED"], [cam[0] - T_approx * 5, cam[-1] + T_approx * 5]
        )
        LED_power = get_LED_powers(ephys_dict["LED"][idx1:idx2], cam, T_approx)

        # return LED and vm on corect segment
        idx1, idx2 = ef.time_to_idx(ephys_dict["LED"], [cam[0] - T_approx, cam[-1]])
        LED = ephys_dict["LED"][idx1:idx2]

        idx1, idx2 = ef.time_to_idx(ephys_dict["vcVm"], [cam[0] - T_approx, cam[-1]])
        vcVm = ephys_dict["vcVm"][idx1:idx2]

        if LED_power[0] < LED_power[1]:
            blue = 0
        else:
            blue = 1

    else:

        blue = 0
        cam = None
        LED = None
        LED_power = None
        vcVm = None
        ephys_fname = None

    if nofilt:
        ratio_stack = stack2rat_nofilt(stack, blue=blue)
    else:
        ratio_stack = stack2rat(stack, blue=blue, causal=washin)

    result_dict = {
        "cam": cam,
        "LED": LED,
        "im": np.mean(stack[blue:100:2], 0),
        "LED_powers": LED_power,
        "ratio_stack": ratio_stack,
        "vcVm": vcVm,
        "tif_file": stack_fname,
        "smr_file": ephys_fname,
    }

    return result_dict


def stack2rat(
    stack, blue=0, av_len=1000, remove_first=True, causal=False, offset=90 * 16
):
    stack -= offset  # remove dark offset

    if remove_first:
        stack = stack[2:, ...]

    if len(stack) % 2 == 1:
        stack = stack[:-1, ...]

    if blue == 0:
        blue = stack[::2, ...].astype(float)
        green = stack[1::2, ...].astype(float)
    else:  # if the leds flipped
        blue = stack[1::2, ...].astype(float)
        green = stack[::2, ...].astype(float)

    if causal:
        origin = (
            av_len // 2 - 1,
            0,
            0,
        )  # whether the filter is centered over the sample, or average strictly previous
    else:
        origin = (0, 0, 0)
    # divide by mean
    blue /= ndimage.uniform_filter(blue, (av_len, 0, 0), mode="nearest", origin=origin)
    green /= ndimage.uniform_filter(
        green, (av_len, 0, 0), mode="nearest", origin=origin
    )

    rat = blue / green

    return rat


def stack2rat_nofilt(stack, blue=0, remove_first=True):
    if remove_first:
        stack = stack[2:, ...]

    if len(stack) % 2 == 1:
        stack = stack[:-1, ...]

    if blue == 0:
        blue = stack[::2, ...].astype(float)
        green = stack[1::2, ...].astype(float)
    else:  # if the leds flipped
        blue = stack[1::2, ...].astype(float)
        green = stack[::2, ...].astype(float)

    blue = get_lin_norm(blue, 2000)
    green = get_lin_norm(green, 2000)

    rat = blue / green

    return rat


def get_lin_norm(st, n):
    slopes = (st[-1, ...] - st[0, ...]) / st.shape[0]
    intercept = st[0, ...]
    bck = slopes * np.arange(st.shape[0])[:, None, None] + intercept
    return st / bck


def strdate2int(strdate):
    return int(strdate[:4]), int(strdate[4:6]), int(strdate[-2:])


def select_daterange(str_date, str_mindate, str_maxdate):
    if (
        (
            datetime.date(*strdate2int(str_date))
            - datetime.date(*strdate2int(str_mindate))
        ).days
        >= 0
    ) and (
        (
            datetime.date(*strdate2int(str_date))
            - datetime.date(*strdate2int(str_maxdate))
        ).days
        <= 0
    ):
        return True
    else:
        return False


def get_tif_smr(
    topdir, savefile, min_date, max_date, prev_sorted=None, only_long=False
):
    if min_date is None:
        min_date = "20000101"
    if max_date is None:
        max_date = "21000101"

    home = Path.home()
    local_home = "/home/peter"
    hpc_home = "/rds/general/user/peq10/home"
    if str(home) == hpc_home:
        HPC = True
    else:
        HPC = False

    files = Path(topdir).glob("./**/*.tif")
    tif_files = []
    smr_files = []

    for f in files:
        
        parts = parts(f)
        day_idx = parts.index('cancer') + 1
        day = re.sub('\D','',parts[day_idx])

        # reject non-date experiment (test etc.)
        try:
            int(day)
        except ValueError:
            continue

        if not select_daterange(day, min_date, max_date):
            continue

        if "long" not in str(f):
            continue

        tif_files.append(str(f))

        
        if str(day)[:4] == '2022':
            #amanda and yilin saved in different format
            smr = [x for x in f.parents[1].glob('*.smr')]
        else:
            # search parents for smr file from deepest to shallowest
            start = f.parts.index(day)
            for i in range(len(f.parts) - 1, start + 1, -1):
                direc = Path(*f.parts[:i])
                smr = [f for f in direc.glob("*.smr")]
                if len(smr) != 0:
                    break

        smr_files.append([str(s) for s in smr])

    max_len = max([len(x) for x in smr_files])

    df = pd.DataFrame()

    df["tif_file"] = tif_files

    for i in range(max_len):
        files = []
        for j in range(len(smr_files)):
            try:
                files.append(smr_files[j][i])
            except IndexError:
                files.append(np.NaN)

        df[f"SMR_file_{i}"] = files

    # now consolidate files that were split (i.e. two tif files in same directory, one has _1 at end,
    # due to file size limits on tif file size)
    remove = []
    for data in df.itertuples():
        fname = data.tif_file
        fname2 = fname[: fname.find(".ome.tif")] + "_1.ome.tif"
        if Path(fname2).is_file():
            df.loc[data.Index, "multi_tif"] = 1
            remove.append(df[df.tif_file == fname2].index[0])
            if Path(fname[: fname.find(".ome.tif")] + "_2.ome.tif").is_file():
                raise NotImplementedError("Not done more than one extra")
        else:
            df.loc[data.Index, "multi_tif"] = 0

    df = df.drop(labels=remove)

    if prev_sorted is not None:
        prev_df = pd.read_csv(prev_sorted)
        if local_home in prev_df.iloc[0].tif_file and HPC:
            mismatch = True
            root = str(Path(hpc_home, "firefly_link"))
            print("mismatch")
        elif hpc_home in prev_df.iloc[0].tif_file and not HPC:
            mismatch = True
            root = str(Path(hpc_home, "data/Firefly"))
        else:
            mismatch = False

        for data in prev_df.itertuples():
            if mismatch:
                tf = data.tif_file
                tf = str(Path(root, tf[tf.find("/cancer/") + 1 :]))
                loc = df[df.tif_file == tf].index
            else:
                loc = df[df.tif_file == data.tif_file].index

            for i in range(max_len):
                if i == 0:
                    if mismatch:
                        sf = data.SMR_file
                        try:
                            sf = str(Path(root, sf[sf.find("/cancer/") + 1 :]))
                        except AttributeError:
                            sf = np.NaN
                        df.loc[loc, f"SMR_file_{i}"] = sf
                    else:
                        df.loc[loc, f"SMR_file_{i}"] = data.SMR_file
                else:
                    df.loc[loc, f"SMR_file_{i}"] = np.NaN

    if only_long:
        df = df[["long_acq" in f for f in df.tif_file]]

    try:
        if np.all(np.isnan(df.SMR_file_1.values.astype(float))):

            df["SMR_file"] = df.SMR_file_0
            for i in range(max_len):
                df = df.drop(columns=f"SMR_file_{i}")
    except Exception:
        pass

    df.to_csv(savefile)

    return df
