__author__ = 'kim'

import numpy as np

def sigma_clipping(date, mag, mag_error, threshold=3, iteration=1):
    """
    Remove any fluctuated data points by magnitudes.

    :param date: An array of dates.
    :param mag: An array of magnitudes.
    :param mag_error: An array of magnitude errors.
    :param threshold: Threshold for sigma-clipping.
    :param iteration: The number of iteration.
    :return: sigma-clipped array of date, mag, mag_error.
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

    plt.errorbar(date, mag, color='b', yerr=err, ls='None')

    start = time.time()
    date, mag, err = \
        sigma_clipping(date, mag, err, threshold=3, iteration=3)
    print '#Processing time:', time.time() - start

    plt.errorbar(date, mag, color='r', yerr=err, ls='None')
    plt.show()