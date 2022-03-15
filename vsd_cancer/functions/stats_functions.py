#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:57:09 2021

@author: peter
"""
import astropy.stats as ass
import numpy as np
import matplotlib.pyplot as plt
import warnings

import numbers

def construct_CI(array,level,function = np.mean,num_resamplings = 10000):
    
    resamplings = ass.bootstrap(array,bootnum = num_resamplings,samples = None, bootfunc = function) 
        
    #todo - check about percentile vs other bootstrap
    CI = np.percentile(resamplings,level/2),np.percentile(resamplings,100-level/2)
    
    return CI, resamplings

def bootstrap_test(control,test,function = np.mean,num_resamplings = 10000,plot = False, names = None):
    '''
    test one sided hypothesis that function(control) > function(test)

    '''
    
    null = np.concatenate([control,test])
    
    control_resamp = ass.bootstrap(null,samples = len(control), bootnum = num_resamplings,bootfunc = function)
    test_resamp = ass.bootstrap(null,samples = len(test),bootnum = num_resamplings,bootfunc = function)


    diff = test_resamp - control_resamp
    res = function(test) - function(control)
    
    pvalue = len(diff[diff < res])/len(diff)
    
    if len(diff[diff < res]) == 0:
        warnings.warn('Not enough resamples to resolve p this small')
        pvalue = 1/num_resamplings #we can only resolve to be up to this val
    
    if plot:
        fig, ax = plt.subplots()
        a = ax.hist(diff,bins = 50, label = 'Resampled Null differences')
        if not len(diff[diff < res]) == 0:
            ax.plot([res,res],[0,a[0].max()], label  = f'Observed difference (p = {pvalue})')
        else:
            ax.plot([res,res],[0,a[0].max()], label  = f'Observed difference (p < {pvalue}, floored)')
        if names is not None:
            ax.set_xlabel(f'Mean {names[1]} - {names[0]}')
        plt.legend(frameon = False)
        
        
        
        return pvalue, diff,fig
    else:
        return pvalue,diff
    
def bootstrap_test_2sided(control,test,function = np.mean,num_resamplings = 10000,plot = False, names = None):
    '''
    test one sided hypothesis that function(control) > function(test)

    '''
    raise NotImplementedError('This is not implemented')
    null = np.concatenate([control,test])
    
    control_resamp = ass.bootstrap(null,samples = len(control), bootnum = num_resamplings,bootfunc = function)
    test_resamp = ass.bootstrap(null,samples = len(test),bootnum = num_resamplings,bootfunc = function)


    diff = test_resamp - control_resamp
    res = function(test) - function(control)
    
    pvalue = len(diff[np.abs(diff) < np.abs(res)])/len(diff)
    
    if len(diff[np.abs(diff) < np.abs(res)]) == 0:
        warnings.warn('Not enough resamples to resolve p this small')
        pvalue = 1/num_resamplings #we can only resolve to be up to this val
    
    if plot:
        fig, ax = plt.subplots()
        a = ax.hist(diff,bins = 50, label = 'Resampled Null differences')
        if not len(diff[np.abs(diff) < np.abs(res)]) == 0:
            ax.plot([res,res],[0,a[0].max()], label  = f'Observed difference (p = {pvalue})')
        else:
            ax.plot([res,res],[0,a[0].max()], label  = f'Observed difference (p < {pvalue}, floored)')
        if names is not None:
            ax.set_xlabel(f'Mean {names[1]} - {names[0]}')
        plt.legend(frameon = False)
        
        
        
        return pvalue, diff,fig
    else:
        return pvalue,diff


def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False,
           rug_length=0.05, rug_kwargs=None, **kwargs):
    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
        
    https://stats.stackexchange.com/questions/403652/two-sample-quantile-quantile-plot-in-python
    """
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, **kwargs)
