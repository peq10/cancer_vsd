import numpy as np
from skimage import io

def read_image(fn, normalize=True):
    """Read a CCD/CMOS image in .da format (Redshirt). [1_]

    Parameters
    ----------
    fn : string
        The input filename.

    Returns
    -------
    images : array, shape (nrow, ncol, nframes)
        The images (normalized by the dark frame if desired).
    frame_interval : float
        The time elapsed between frames, in milliseconds.
    bnc : array, shape (8, nframes)
        The bnc data.
    dark_frame : array, shape (nrow, ncol)
        The dark frame by which the image data should be normalized.

    Notes
    -----
    Interlaced images, as produced by the option "write directly to disk",
    are not currently supported.

    References
    ----------
    .. [1] http://www.redshirtimaging.com/support/dfo.html
    """
    data = np.fromfile(fn,dtype = np.int16)
    header_size = 2560
    header = data[:header_size]
    ncols, nrows = map(int, header[384:386])  # prevent int16 overflow
    nframes = int(header[4])
    frame_interval = header[388] / 1000
    acquisition_ratio = header[391]
    if frame_interval >= 10:
        frame_interval *= header[390]  # dividing factor
    image_size = nrows * ncols * nframes
    bnc_start = header_size + image_size
    images = np.reshape(np.array(data[header_size:bnc_start]),
                        (nrows, ncols, nframes))
    bnc_end = bnc_start + 8 * acquisition_ratio * nframes
    bnc = np.reshape(np.array(data[bnc_start:bnc_end]), (8, nframes * acquisition_ratio))
    dark_frame = np.reshape(np.array(data[bnc_end:-8]), (nrows, ncols))
    if normalize:
        images -= dark_frame[..., np.newaxis]
    return np.rollaxis(images,-1), frame_interval, bnc, dark_frame


def convert_images(fns, normalize=True):
    for fn in fns:
        image, frame_interval, bnc, dark_frame = read_image(fn, normalize)
        out_fn = fn[:-3] + '.tif'
        out_fn_dark = fn[:-3] + '.dark_frame.tif'
        io.imsave(out_fn, np.transpose(image, (2, 0, 1)),
                  plugin='tifffile', compress=1)
        io.imsave(out_fn_dark, dark_frame, plugin='tifffile', compress=1)
