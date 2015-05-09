__author__ = 'kim'

import numpy as np


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


def sigma_clipping(date, mag, mag_error, threshold=3, iteration=1):
    """
    Remove any fluctuated data points by magnitudes.

    :param date: An array of dates.
    :param mag: An array of magnitudes.
    :param mag_error: An array of magnitude errors.
    :param threshold: Threshold for sigma-clipping.
    :param iteration: The number of iteration.
    :return: Sigma-clipped arrays of date, mag, and mag_error.
    """

    # By magnitudes
    for i in range(int(iteration)):
        mean = np.median(mag)
        std = np.std(mag)

        index = (mag >= mean - threshold*std) & (mag <= mean + threshold*std)
        date = date[index]
        mag = mag[index]
        mag_error = mag_error[index]

    '''
    # By magnitude errors
    for i in range(int(iteration)):
        mean = np.median(mag_error)
        std = np.std(mag_error)

        index = (mag_error >= mean - threshold*std) & \
            (mag_error <= mean + threshold*std)
        date = date[index]
        mag = mag[index]
        mag_error = mag_error[index]
    '''

    return date, mag, mag_error


def plot_folded_lc(date, mag, features, values,
        output_folder='./output/', filename='./output.png'):
    """
    Generate a plot of a phase-folded light-curve for a given period.

    :param date: An array of dates.
    :param mag: An array of magnitudes.
    :param features: A list of features' name, returned from UPSILoN.
    :param values: A list of features' value, returned from UPSILoN.
    :param output_folder: An output folder for a generated image.
    :param filename: An output filename for a generated image.
    :return:
    """

    index = np.where(features=='period')
    plot_folded_lc(date, mag, values[index], output_folder)


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
    import matplotlib.pyplot as plt

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

    data = np.loadtxt('./datasets/lightcurves/lm0010n22323.time')
    date = data[:, 0]
    mag = data[:, 1]
    err = data[:, 2]

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

