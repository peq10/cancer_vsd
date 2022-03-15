def cc(times_spikes_pre, times_spikes_post, binsize):
    """ Calculates the Pearson's correlation coefficient between two spike trains
    Inputs are:
    times_spikes_pre - array with time points in which the presynaptic neuron fired
    times_spikes_post - array with time points in which the postsynaptic neuron fired
    binsize - size of the bins for constructing spike trains from spike times (float)
    Returns:
    cc - Pearson's correlation coefficient between both spike trains (float)
    * The spike trains are created with a common time array that expand the overall minimum and maximum time stamp given.
    * Therefore, only time stamps on the window of interest should be passed to the function.
    """
    time_bins = np.arange(
        min([min(times_spikes_pre), min(times_spikes_post)]),
        max([max(times_spikes_pre), max(times_spikes_post)]) + binsize,
        binsize,
    )
    spk_train_source = np.histogram(times_spikes_pre, bins=time_bins)[0]
    spk_train_target = np.histogram(times_spikes_post, bins=time_bins)[0]
    spk_train_source[spk_train_source > 1] = 1
    spk_train_target[spk_train_target > 1] = 1
    cc = np.corrcoef(spk_train_source, spk_train_target)[0, 1]
    return cc
