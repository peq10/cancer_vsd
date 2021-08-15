#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:09:07 2021

@author: peter
"""
#a script to get all the events

import numpy as np

import pandas as pd
from pathlib import Path

from vsd_cancer.functions import cancer_functions as canf

import matplotlib.pyplot as plt


def get_all_powers(initial_df,save_dir):

    df = pd.read_csv(initial_df)
    
    blue_cal = np.load(Path(save_dir,'LED_calibration_20201113_blue.npy')) # saved as slope,intercept
    green_cal = np.load(Path(save_dir,'LED_calibration_20201113_blue.npy'))
    
    all_blue = []
    all_green = []
    trials = []

    for idx, data in enumerate(df.itertuples()):
        
        if data.date > 20210121 or not data.use:
            continue
        
        
        if data.date >= 20201228:
            blue_cal = np.load(Path(save_dir,'LED_calibration_20201113_blue.npy')) # saved as slope,intercept
            green_cal = np.load(Path(save_dir,'LED_calibration_20201113_blue.npy'))
        else:
            blue_cal = np.load(Path(save_dir,'LED_calibration_20201228_blue.npy')) # saved as slope,intercept
            green_cal = np.load(Path(save_dir,'LED_calibration_20201228_blue.npy'))
        
        trial_string = data.trial_string

        trial_save = Path(save_dir,'ratio_stacks',trial_string)

        LED_powers = np.load(Path(trial_save,f'{trial_string}_LED_powers.npy'))
        
        LED_powers.sort() #blue alwasy less
        
        blue_power = LED_powers[0]*blue_cal[0] + blue_cal[1]
        green_power = LED_powers[1]*green_cal[0] + green_cal[1]
        
        all_blue.append(blue_power)
        all_green.append(green_power)
        trials.append(data.trial_string)
        

    LED_df = pd.DataFrame({'trial':trials,'blue':all_blue,'green':all_green})
    
    LED_df.to_csv(Path(save_dir,'LED_powers_mw_mm2.csv'))
    
    
    