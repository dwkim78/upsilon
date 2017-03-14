"""
Base IO code for all datasets
"""

import sys
import numpy as np

from os.path import dirname
from os.path import join


def load_EROS_lc(filename='lm0010n22323.time'):
    """
    Read an EROS light curve and return its data.

    Parameters
    ----------
    filename : str, optional
        A light-curve filename.

    Returns
    -------
    dates : numpy.ndarray
        An array of dates.
    magnitudes : numpy.ndarray
        An array of magnitudes.
    errors : numpy.ndarray
        An array of magnitudes errors.
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
    Return the UPSILoN random forests classifier.

    The classifier is trained using OGLE and EROS periodic variables
    (Kim et al. 2015).

    Returns
    -------
    clf : sklearn.ensemble.RandomForestClassifier
        The UPSILoN random forests classifier.
    """

    import gzip
    try:
        import cPickle as pickle
    except:
        import pickle

    module_path = dirname(__file__)
    file_path = join(module_path, 'models/rf.model.sub.github.gz')

    # For Python 3.
    if sys.version_info.major >= 3:
        clf = pickle.load(gzip.open(file_path, 'rb'), encoding='latin1')
    # For Python 2.
    else:
        clf = pickle.load(gzip.open(file_path, 'rb'))

    return clf
