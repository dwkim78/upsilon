import numpy as np


def sigma_clipping(date, mag, err, threshold=3, iteration=1):
    """
    Remove any fluctuated data points by magnitudes.

    Parameters
    ----------
    date : array_like
        An array of dates.
    mag : array_like
        An array of magnitudes.
    err : array_like
        An array of magnitude errors.
    threshold : float, optional
        Threshold for sigma-clipping.
    iteration : int, optional
        The number of iteration.

    Returns
    -------
    date : array_like
        Sigma-clipped dates.
    mag : array_like
        Sigma-clipped magnitudes.
    err : array_like
        Sigma-clipped magnitude errors.
    """

    # Check length.
    if (len(date) != len(mag)) \
        or (len(date) != len(err)) \
        or (len(mag) != len(err)):
        raise RuntimeError('The length of date, mag, and err must be same.')

    # By magnitudes
    for i in range(int(iteration)):
        mean = np.median(mag)
        std = np.std(mag)

        index = (mag >= mean - threshold*std) & (mag <= mean + threshold*std)
        date = date[index]
        mag = mag[index]
        err = err[index]

    return date, mag, err
