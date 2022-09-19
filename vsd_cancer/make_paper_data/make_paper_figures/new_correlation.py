#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 17:38:55 2022

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt

null_dists_resamples = np.load(
    "/media/peter/bigdata/Firefly/cancer/analysis/full/correlation/null_dists.npy"
)
bootstrapped_resamples = np.load(
    "/media/peter/bigdata/Firefly/cancer/analysis/full/correlation/bootstrapped_samples.npy"
)

binsizes = np.load(
    "/media/peter/bigdata/Firefly/cancer/analysis/full/correlation/binsizes.npy"
)

binsizes2 = binsizes[:, None] * np.ones((1, 1000))

a = plt.hist2d(binsizes2.ravel(), null_dists_resamples.ravel(), bins=(20, 100))

b = plt.hist2d(binsizes2.ravel(), bootstrapped_resamples.ravel(), bins=(20, 100))
