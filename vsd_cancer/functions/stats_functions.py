#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:57:09 2021

@author: peter
"""
import astropy.stats as ass
import numpy as np
import matplotlib.pyplot as plt

def construct_CI(array,level,function = np.mean,num_resamplings = 10000):
    
    resamplings = ass.bootstrap(array,bootnum = num_resamplings,samples = None, bootfunc = function) 
        
    #todo - check about percentile vs other bootstrap
    CI = np.percentile(resamplings,level/2),np.percentile(resamplings,100-level/2)
    
    return CI, resamplings

def bootstrap_test(control,test,function = np.mean,num_resamplings = 10000,plot = False):
    '''
    test one sided hypothesis that function(control) > function(test)

    '''
    
    null = np.concatenate([control,test])
    
    control_resamp = ass.bootstrap(null,samples = len(control), bootnum = num_resamplings,bootfunc = function)
    test_resamp = ass.bootstrap(null,samples = len(test),bootnum = num_resamplings,bootfunc = function)


    diff = test_resamp - control_resamp
    res = function(test) - function(control)
    
    pvalue = len(diff[diff < res])/len(diff)
    
    if plot:
        plt.figure()
        a = plt.hist(diff,bins = 50)
        plt.plot([res,res],[0,a[0].max()])
        
    return pvalue, diff