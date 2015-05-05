__author__ = 'kim'

"""
Base IO code for all datasets
"""

import numpy as np

from os.path import dirname
from os.path import join


def load_EROS_lc(filename='lm0010n22323.time'):
    """
    Read an EROS light curve and return its data.

    :param filename: A light-curve filename.
    :return: Arrays of dates, magnitudes, and magnitudes errors.
    """

    module_path = dirname(__file__)
    file_path = join(module_path, 'lightcurves', filename)

    data = np.loadtxt(file_path)
    date = data[:, 0]
    mag = data[:, 1]
    err = data[:, 2]

    return date, mag, err


