__author__ = 'kim'

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

import period_LS_pyfftw as pLS

class ExtractFeatures():
    """
    Extract variability features of a light curve.
    """

    def __init__(self, date, mag, err=None, n_threads=4):
        """
        Initialize.

        :param date: An array of observed date, in days.
        :param mag: An array of observed magnitude.
        :param err: An array of magnitude error. If None, std(mag) will be used.
        :param n_threads: The number of cores to use to derive periods.
        :return: An array of variability features.
        """

        # Set basic values.
        self.date = date
        self.mag = mag
        if err is not None:
            self.err = err
        else:
            self.err = np.ones(len(self.mag)) * np.std(self.mag)

        self.n_threads = n_threads

    def sallow_run(self):
        """
        Derive not-period-based features.

        :return: None
        """
        # Number of data points before/after filtering.
        self.n_points_raw = 0
        self.n_points_filter = 0

        # Weight calculation
        self.weight = 1. / self.err
        self.weighted_sum = np.sum(self.weight)

        # Simple statistics, mean, median and std.
        self.mean = np.mean(self.mag)
        self.median = np.median(self.mag)
        self.std = np.std(self.mag)

        # Weighted mean and std.
        self.weighted_mean = np.sum(self.mag * self.weight) / self.weighted_sum
        self.weighted_std = np.sqrt(np.sum((self.mag - self.weighted_mean)**2 \
            * self.weight) / self.weighted_sum)

        # Skewness and kurtosis.
        self.skewness = ss.skew(self.mag)
        self.kurtosis = ss.kurtosis(self.mag)

        # Normalization-test. Shapiro-Wilk test.
        shapiro = ss.shapiro(self.mag)
        self.shapiro_W = shapiro[0]
        self.shapiro_log10p = np.log10(shapiro[1])

        # Percentile features.
        self.quartile31 = np.percentile(self.mag, 75) \
            - np.percentile(self.mag, 25)

        # Stetson K.
        self.stetsonK = self.get_stetson_k(self.mag, self.median, self.err)

        # Ratio between higher and lower amplitude than average.
        self.hl_amp_ratio = self.half_mag_amplitude_ratio(
            self.mag, self.median, self.weight)

    def deep_run(self):
        """
        Derive period-based features.

        :return: None
        """
        # Lomb-Scargle period finding.
        self.getPeriodLS(self.date, self.mag)

        # Features based on a phase-folded light curve
        # such as Eta, slope-percentile, etc.
        # Should be called after the getPeriodLS() is called.

        # Created phased a folded light curve.
        # We use period * 2 to take eclipsing binaries into account.
        phase_folded_date = self.date % (self.period * 2.)
        sorted_index = np.argsort(phase_folded_date)

        folded_date = phase_folded_date[sorted_index]
        folded_mag = self.mag[sorted_index]

        # phase Eta
        self.phase_eta = self.Eta(folded_mag, np.std(folded_mag))

        # Slope percentile.
        self.slope_per10, self.slope_per90 = \
            self.slope_percentile(folded_date, folded_mag)


    def getPeriodLS(self, date, mag):
        """
        Period finding using the Lomb-Scargle algorithm.

        Finding two periods. The second period is estimated
        after whitening the first period.

        Calculating various other features as well using derived periods.

        :param date: An array of observed date, in days.
        :param mag: An array of observed magnitude.
        """

        # DO NOT CHANGE THESE PARAMETERS.
        oversampling = 3.
        min_period = 0.03 # in days
        hifac = int((max(date) - min(date)) / len(date) / min_period * 2.)

        # Minimum hifac
        if hifac < 100:
            hifac = 100

        #------------------------------------------------------------------#
        # IF YOU DEADLY NEED TO SPEED UP, UNCOMMENT BELOW THREE LINES.     #
        #------------------------------------------------------------------#
        #oversampling = 2.
        #min_period = 0.1
        #hifac = 100
        #------------------------------------------------------------------#
        # THIS MIGHT MAKES THE Lomb-Scargle 200 ~ 300% FASTER.             #
        # BUT THIS MIGHT NOT DEGRADE SHORT-PERIOD VARIABLE CLASSIFICATION. #
        #------------------------------------------------------------------#

        # Lomb-Scargle.
        fx, fy, nout, jmax, prob = pLS.fasper(date, mag, oversampling, hifac)

        self.f = fx[jmax]
        self.period = 1. / self.f
        self.f_log10FAP = \
            np.log10(pLS.getSignificance(fx, fy, nout, oversampling)[jmax])
        self.f_SNR1 = fy[jmax] / np.median(fy)
        self.f_SNR2 = (fy[jmax] - np.median(fy)) / np.std(fy)

        # Fit Fourier Series of order 5.
        order = 5
        # Initial guess of Fourier coefficients.
        p0 = np.ones(order * 2 + 1)
        date_period = (date % self.period) / self.period
        p1, success = leastsq(self.residuals, p0,
            args=(date_period, mag, order))
        #fitted_y = self.FourierSeries(p1, date_period, order)
        #print p1

        #print p1, self.mean, self.median
        #plt.plot(date_period, self.mag, 'b+')
        #plt.show()

        # Derive Fourier features for the first period.
        #Petersen, J. O., 1986, A&A
        self.f_amp = 2. * np.sqrt(p1[1]**2 + p1[2]**2)
        self.f_R21 = np.sqrt(p1[3]**2 + p1[4]**2) / self.f_amp
        self.f_R31 = np.sqrt(p1[5]**2 + p1[6]**2) / self.f_amp
        self.f_R41 = np.sqrt(p1[7]**2 + p1[8]**2) / self.f_amp
        self.f_R51 = np.sqrt(p1[9]**2 + p1[10]**2) / self.f_amp
        self.f_phase = np.arctan(-p1[1] / p1[2])
        self.f_phi21 = np.arctan(-p1[3] / p1[4]) - 2. * self.f_phase
        self.f_phi31 = np.arctan(-p1[5] / p1[6]) - 3. * self.f_phase
        self.f_phi41 = np.arctan(-p1[7] / p1[8]) - 4. * self.f_phase
        self.f_phi51 = np.arctan(-p1[9] / p1[10]) - 5. * self.f_phase

        """
        # Derive the second period
        # Whitening a light curve.
        residual_mag = mag - fitted_y

        # Lomb-Scargle again to find the second period.
        omega_top, power_top = search_frequencies(date, residual_mag, err,
            #LS_kwargs={'generalized':True, 'subtract_mean':True},
            n_eval=5000, n_retry=3, n_save=50)

        self.period2 = 2*np.pi/omega_top[np.where(power_top==np.max(power_top))][0]
        self.f2 = 1. / self.period2
        self.f2_SNR = power_top[np.where(power_top==np.max(power_top))][0] \
            * (len(self.date) - 1) / 2.

        # Fit Fourier Series again.
        p0 = [1.] * order * 2
        date_period = (date % self.period) / self.period
        p2, success = leastsq(self.residuals, p0,
            args=(date_period, residual_mag, order))
        fitted_y = self.FourierSeries(p2, date_period, order)

        #plt.plot(date%self.period2, residual_mag, 'b+')
        #plt.show()

        # Derive Fourier features for the first second.
        self.f2_amp = 2. * np.sqrt(p2[1]**2 + p2[2]**2)
        self.f2_R21 = np.sqrt(p2[3]**2 + p2[4]**2) / self.f2_amp
        self.f2_R31 = np.sqrt(p2[5]**2 + p2[6]**2) / self.f2_amp
        self.f2_R41 = np.sqrt(p2[7]**2 + p2[8]**2) / self.f2_amp
        self.f2_R51 = np.sqrt(p2[9]**2 + p2[10]**2) / self.f2_amp
        self.f2_phase = np.arctan(-p2[1] / p2[2])
        self.f2_phi21 = np.arctan(-p2[3] / p2[4]) - 2. * self.f2_phase
        self.f2_phi31 = np.arctan(-p2[5] / p2[6]) - 3. * self.f2_phase
        self.f2_phi41 = np.arctan(-p2[7] / p2[8]) - 4. * self.f2_phase
        self.f2_phi51 = np.arctan(-p2[9] / p2[10]) - 5. * self.f2_phase

        # Calculate features using the first and second periods.
        self.f12_ratio = self.f2 / self.f1
        self.f12_remain = self.f1 % self.f2 \
            if self.f1 > self.f2 else self.f2 % self.f1
        self.f12_amp = self.f2_amp / self.f1_amp
        self.f12_phase = self.f2_phase - self.f1_phase
        """

    def residuals(self, pars, x, y, order):
        """
        Residual of FourierSeries.

        :param pars: Fourier series parameters.
        :param x: An array of date.
        :param y: An array of true values to fit.
        :param order: order of Fourier Series.
        """

        return y - self.FourierSeries(pars, x, order)

    def FourierSeries(self, pars, x, order):
        """
        Function to fit Fourier Series.

        :param x: An array of date divided by period. It doesn't need to be sorted.
        :param pars: Fourier series parameters.
        :param order: An order of Fourier series.
        """

        sum = pars[0]
        for i in range(order):
            sum += pars[i * 2 + 1] * np.sin(2 * np.pi * (i + 1) * x) \
                + pars[i * 2 + 2] * np.cos(2 * np.pi * (i + 1) * x)

        return sum

    def get_stetson_k(self, mag, avg, err):
        """
        Return Stetson K feature.

        :param mag: An array of magnitude.
        :param avg: An average value of magnitudes.
        :param err: An array of magnitude errors.
        :return: Stetson K value.
        """

        residual = (mag - avg) / err
        stetsonK = np.sum(np.fabs(residual)) \
            / np.sqrt(np.sum(residual * residual)) / np.sqrt(len(mag))

        return stetsonK

    def half_mag_amplitude_ratio(self, mag, avg, weight):
        """
        Return ratio of amplitude of higher and lower
        magnitudes than average.

        This ratio, by definition, should be higher for EB
        than for others.

        :param mag: An array of magnitudes.
        :param avg: An average value of magnitudes.
        :param weight: array of weight.
        :return: Ratio of amplitude of higher and lower magnitudes than average.
        """

        # For lower (fainter) magnitude than average.
        index = np.where(mag > avg)
        lower_weight = weight[index]
        lower_weight_sum = np.sum(lower_weight)
        lower_mag = mag[index]

        lower_mag_weighted_mean = np.sum(lower_mag * lower_weight) / \
            lower_weight_sum
        lower_weighted_std = np.sqrt(np.sum((lower_mag
            - lower_mag_weighted_mean)**2
            * lower_weight) / lower_weight_sum)

        # For higher (brighter) magnitude than average.
        index = np.where(mag <= avg)
        higher_weight = weight[index]
        higher_weight_sum = np.sum(higher_weight)
        higher_mag = mag[index]

        higher_mag_weighted_mean = np.sum(higher_mag * higher_weight) / \
            higher_weight_sum
        higher_weighted_std = np.sqrt(np.sum((higher_mag
            - higher_mag_weighted_mean)**2
            * higher_weight) / higher_weight_sum)

        # Return ratio.
        return lower_weighted_std / higher_weighted_std

    def Eta(self, mag, std):
        """
        Return Eta feature.

        :param mag: An array of magnitudes.
        :param std: std of magnitudes.
        :return: Eta
        """

        diff = mag[1:] - mag[:len(mag) - 1]
        eta = np.sum(diff * diff) / (len(mag) - 1.) / std / std

        return eta


    def slope_percentile(self, date, mag):
        """
        Return 10% and 90% percentile of slope.

        :param date: An array of phase-folded date. Sorted.
        :param mag: An array of phase-folded magnitudes. Sorted by date.
        :return: 10% and 90% percentile values of slope.
        """

        date_diff = date[1:] - date[:len(date) - 1]
        mag_diff = mag[1:] - mag[:len(mag) - 1]

        # Remove zero mag_diff.
        index = np.where(mag_diff != 0.)
        date_diff = date_diff[index]
        mag_diff = mag_diff[index]

        # Derive slope.
        slope = date_diff / mag_diff

        percentile_10 = np.percentile(slope, 10.)
        percentile_90 = np.percentile(slope, 90.)

        return percentile_10, percentile_90

    def get_features(self):
        """
        Return all features with its names.

        :return: Feature names, feature values
        """

        feature_names = []
        feature_values = []

        # Get all the names of features.
        all_vars = vars(self)
        for name in all_vars.keys():
            # Omit input variables such as date, mag, err, etc.
            if not (name == 'date' or name == 'mag' or name == 'err'
                    or name == 'weight' or name == 'weighted_sum'
                    or name == 'n_threads'):
                # Filter some other unnecessary features
                if not (name == 'f1' or name == 'f2'):
                    feature_names.append(name)

        # Sort by the names.
        # Sorting should be done to keep maintaining the same order of features.
        feature_names.sort()

        # Get feature values.
        for name in feature_names:
            feature_values.append(all_vars[name])

        return feature_names, feature_values
