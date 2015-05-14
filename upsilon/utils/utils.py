__author__ = 'kim'

import numpy as np
from upsilon.datasets.base import load_EROS_lc


def sigma_clipping_without_error(date, mag, threshold=3, iteration=1):
    """
    Remove any fluctuated data points by magnitudes.

    :param date: An array of dates.
    :param mag: An array of magnitudes.
    :param threshold: Threshold for sigma-clipping.
    :param iteration: The number of iteration.
    :return: Sigma-clipped arrays of date, mag, and mag_error.
    """

    mag_error = np.ones(len(date)) * np.std(mag)
    return sigma_clipping(date, mag, mag_error, threshold, iteration)


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


def plot_folded_lc(date, mag, features,
        output_folder='./output/', filename='./output.png'):
    """
    Generate a plot of a phase-folded light-curve for a given period.

    :param date: An array of dates.
    :param mag: An array of magnitudes.
    :param features: Features, returned from UPSILoN.
    :param output_folder: An output folder for a generated image.
    :param filename: An output filename for a generated image.
    :return:
    """

    plot_folded_lc(date, mag, features['period'], output_folder)


def __plot_folded_lc(date, mag, period, output_folder='./output/',
        filename='./output.png'):
    """
    Generate a plot of a phase-folded light-curve for a given period.

    :param date: An array of dates.
    :param mag: An array of magnitudes.
    :param mag_error: An array of magnitude errors.
    :param period: A period estimated by UPSILoN.
    :param output_folder: An output folder for a generated image.
    :param filename: An output filename for a generated image.
    :return:
    """

    import os
    import sys
    try:
        import matplotlib.pyplot as plt
    except:
        raise RuntimeError('No Matplotlib detected')

    phase_date = (date % period) / period
    plt.errorbar(phase_date, mag,  color='k', marker='.',
        ls='None')
    plt.title('Period: %.5f days' % period, fontsize=15)
    plt.xlabel('Phase', fontsize=15)
    plt.ylabel('Magnitude', fontsize=15)
    ax = plt.gca()
    ax.invert_yaxis()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + '/' + filename)


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    date, mag, err = load_EROS_lc()

    index = mag < 99.999
    date = date[index]
    mag = mag[index]
    err = err[index]
    #plt.errorbar(date, mag, color='b', yerr=err, ls='None')

    start = time.time()
    date, mag, err = \
        sigma_clipping(date, mag, err, threshold=3, iteration=3)
    print '#Processing time:', time.time() - start
    #plt.errorbar(date, mag, color='r', yerr=err, ls='None')
    #plt.show()

    __plot_folded_lc(date, mag, 4.19044052846)

