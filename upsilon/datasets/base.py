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


def load_rf_model():
    """
    Return the UPSILoN random forests classifier trained using OGLE and
    EROS periodic variables (Kim et al. 2015).
    :return: the UPSILoN random forests classifier.
    """

    import gzip
    try:
        import cPickle as pickle
    except:
        import pickle

    module_path = dirname(__file__)
    file_path = join(module_path, 'models/rf.model.sub.github.gz')
    clf = pickle.load(gzip.open(file_path, 'rb'))

    return clf