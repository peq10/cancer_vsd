import os

import numpy as np
from scipy import ndimage as ndi
from skimage import io
from skimage.filters import threshold_li
import toolz as tz
from toolz import curried as C
from fn import _ as X

from .read import read_image


def unzip(arg):
    return zip(*arg)


def _extract_roi(image, axis=-1):
    max_frame = np.max(image, axis=axis)
    initial_mask = max_frame > threshold_li(max_frame)
    regions = ndi.label(initial_mask)[0]
    region_sizes = np.bincount(np.ravel(regions))
    return regions == (np.argmax(region_sizes[1:]) + 1)


def extract_trace(image, axis=-1):
    """Get a mean intensity trace over time out of an image.

    Parameters
    ----------
    image : array
        The input image.
    axis : int, optional
        The axis identifying frames.

    Returns
    -------
    trace : array of float
        The trace of the image data over time.
    roi : array of bool
        The mask used to obtain the trace.
    """
    roi = _extract_roi(image, axis)
    trace = np.sum(image[roi].astype(float), axis=0) / np.sum(roi)
    return trace, roi


def process_directory(directory, output_filename='traces.csv'):
    """Extract traces and ROIs for all .da files in a directory.

    Parameters
    ----------
    directory : string
        The directory containing the .da files to be processed.
    output_filename : string
        The name of the file to write the results to.
    """
    filenames = tz.pipe(directory, os.listdir,
                        C.filter(X.call('endswith', '.da')), sorted)
    filenames = [os.path.join(directory, fn) for fn in filenames]
    images, frame_intervals, bncs, dark_frames = unzip(map(read_image,
                                                           filenames))
    traces, rois = unzip(map(extract_trace, images))
    with open(output_filename, 'w') as fout:
        for filename, frame_interval, trace, roi in \
                            zip(filenames, frame_intervals, traces, rois):
            line = ','.join([os.path.basename(filename), str(frame_interval)] +
                            list(map(str, trace)))
            fout.write(line + '\n')
            io.imsave(filename[:-3] + '.roi.tif', roi.astype(np.uint8) * 255,
                      plugin='tifffile', compress=1)
