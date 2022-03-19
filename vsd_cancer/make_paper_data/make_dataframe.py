#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:57:33 2020

@author: peter
"""
from vsd_cancer.functions import cancer_functions as canf
from pathlib import Path
import datetime
import numpy as np

home = Path.home()

if "peq10" in str(home):
    HPC = True
    top_dir = Path(home, "firefly_link/cancer")
    savestr = "_HPC"
else:
    HPC = False
    top_dir = Path(home, "data/Firefly/cancer")
    savestr = ""


save_file = Path(
    top_dir,
    "analysis",
    f"long_acqs_10A_{datetime.datetime.now().year}{datetime.datetime.now().month:02}{datetime.datetime.now().day:02}{savestr}.csv",
)
prev_sorted = Path(top_dir, "analysis", "long_acqs_20201230_sorted.csv")


df = canf.get_tif_smr(
    top_dir, save_file, "20220101", None, prev_sorted=None, only_long=True
)

dates = []
slips = []
areas = []
TTX = []
high_k = []
trial_string = []
n_frames = []
for data in df.itertuples():
    s = data.tif_file

    dates.append(
        s[
            s.find("/cancer/")
            + len("/cancer/") : s.find("/cancer/")
            + len("/cancer/")
            + 8
        ]
    )
    slips.append(s[s.find("slip") + len("slip") : s.find("slip") + len("slip") + 1])

    if "area" in s:
        areas.append(s[s.find("area") + len("area") : s.find("area") + len("area") + 1])
    else:
        areas.append(s[s.find("cell") + len("cell") : s.find("cell") + len("cell") + 1])

    if "ttx" in s.lower():
        TTX.append(True)
    else:
        TTX.append(False)

    if "high_k" in s.lower():
        high_k.append(True)
    else:
        high_k.append(False)

    trial_string.append("_".join(Path(s).parts[Path(s).parts.index("cancer") : -1]))

    try:
        meta = canf.load_tif_metadata(s)
        n_fr = len(meta) - 1
        n_frames.append(n_fr)
    except FileNotFoundError:
        n_fr = -1
        n_frames.append(-1)

    if n_fr != 10000:
        print(s)
        print(len(meta) - 1)

# also expand dataframe


df["date"] = dates
df["slip"] = slips
df["area"] = areas
df["trial_string"] = trial_string
df["ttx"] = TTX
df["high_k"] = high_k
df["n_frames"] = n_frames

# drop bad goes
df = df[df["n_frames"] > 1500]

df = df.sort_values(by=["date", "slip", "area"])

df.to_csv(
    Path(
        top_dir,
        "analysis",
        f"long_acqs_{datetime.datetime.now().year}{datetime.datetime.now().month:02}{datetime.datetime.now().day:02}{savestr}_labelled.csv",
    )
)
