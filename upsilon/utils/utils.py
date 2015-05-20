__author__ = 'kim'

import numpy as np
from upsilon.datasets.base import load_EROS_lc


def sigma_clipping(date, mag, err, threshold=3, iteration=1):
    """
    Remove any fluctuated data points by magnitudes.

    :param date: An array of dates.
    :param mag: An array of magnitudes.
    :param err: An array of magnitude errors.
    :param threshold: Threshold for sigma-clipping.
    :param iteration: The number of iteration.
    :return: Sigma-clipped arrays of date, mag, and mag_error.
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
