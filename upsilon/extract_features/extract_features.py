__author__ = 'kim'

import warnings
import multiprocessing
from collections import OrderedDict

import numpy as np
import scipy.stats as ss
from scipy.optimize import leastsq

import period_LS_pyfftw as pLS

from feature_set import get_feature_set
from feature_set import get_feature_set_all

feature_names_list = get_feature_set()
feature_names_list_all = get_feature_set_all()

class ExtractFeatures():
    """
    Extract variability features of a light curve.
    """

    def __init__(self, date, mag, err=None, n_threads=4, min_period=0.03):
        """
        Initialize.

        :param date: An array of observed date, in days.
        :param mag: An array of observed magnitude.
        :param err: An array of magnitude error. If None, std(mag) will be used.
        :param n_threads: The number of cores to use to derive periods.
        :param min_period: The minimum period to calculate.
        :return: An array of variability features.
        """

        # Set basic values.
        if not isinstance(date, np.ndarray):
            date = np.array(date)
        if not isinstance(mag, np.ndarray):
            mag = np.array(mag)

        self.date = date
        self.mag = mag
        if err is not None:
            if not isinstance(err, np.ndarray):
                err = np.array(err)
            self.err = err
        else:
            self.err = np.ones(len(self.mag)) * np.std(self.mag)

        # Check length.
        if (len(self.date) != len(self.mag)) \
            or (len(self.date) != len(self.err)) \
            or (len(self.mag) != len(self.err)):
            raise RuntimeError('The length of date, mag, and err must be same.')

        # if the number of data points is too small.
        min_n_data = 80
        if len(self.date) < min_n_data:
            warnings.warn('The number of data points are less than %d.'
                % min_n_data)

        n_threads = int(n_threads)
        if n_threads > multiprocessing.cpu_count():
            self.n_threads = multiprocessing.cpu_count()
        else:
            if n_threads <= 0:
                self.n_threads = 1
            else:
                self.n_threads = n_threads

        min_period = float(min_period)
        if min_period <= 0:
            self.min_period = 0.03
        else:
            self.min_period = min_period

    def run(self):
        """
        Run feature extraction modules.
        """

        self.shallow_run()
        self.deep_run()

    def shallow_run(self):
        """
        Derive not-period-based features.

        :return: None
        """
        # Number of data points
        self.n_points = len(self.date)

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
        self.shapiro_w = shapiro[0]
        #self.shapiro_log10p = np.log10(shapiro[1])

        # Percentile features.
        self.quartile31 = np.percentile(self.mag, 75) \
            - np.percentile(self.mag, 25)

        # Stetson K.
        self.stetson_k = self.get_stetson_k(self.mag, self.median, self.err)

        # Ratio between higher and lower amplitude than average.
        self.hl_amp_ratio = self.half_mag_amplitude_ratio(
            self.mag, self.median, self.weight)
        # This second function's value is very similar with the above one.
        #self.hl_amp_ratio2 = self.half_mag_amplitude_ratio2(
        #    self.mag, self.median)

        # Cusum
        self.cusum = self.get_cusum(self.mag)

        # Eta
        self.eta = self.get_eta(self.mag, self.weighted_std)

    def deep_run(self):
        """
        Derive period-based features.

        :return: None
        """
        # Lomb-Scargle period finding.
        self.get_period_LS(self.date, self.mag, self.n_threads, self.min_period)

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
        self.phase_eta = self.get_eta(folded_mag, self.weighted_std)

        # Slope percentile.
        self.slope_per10, self.slope_per90 = \
            self.slope_percentile(folded_date, folded_mag)

        # phase Cusum
        self.phase_cusum = self.get_cusum(folded_mag)

    def get_period_LS(self, date, mag, n_threads, min_period):
        """
        Period finding using the Lomb-Scargle algorithm.

        Finding two periods. The second period is estimated
        after whitening the first period.

        Calculating various other features as well using derived periods.

        :param date: An array of observed date, in days.
        :param mag: An array of observed magnitude.
        :param n_threads: The number of threads to use.
        :param min_period: The minimum period to calculate.
        """

        # DO NOT CHANGE THESE PARAMETERS.
        oversampling = 3.
        hifac = int((max(date) - min(date)) / len(date) / min_period * 2.)

        # Minimum hifac
        if hifac < 100:
            hifac = 100

        # Lomb-Scargle.
        fx, fy, nout, jmax, prob = pLS.fasper(date, mag, oversampling, hifac,
            n_threads)

        self.f = fx[jmax]
        self.period = 1. / self.f
        self.period_uncertainty = self.get_period_uncertainty(fx, fy, jmax)
        self.period_log10FAP = \
            np.log10(pLS.getSignificance(fx, fy, nout, oversampling)[jmax])
        #self.f_SNR1 = fy[jmax] / np.median(fy)
        self.period_SNR = (fy[jmax] - np.median(fy)) / np.std(fy)

        # Fit Fourier Series of order 3.
        order = 3
        # Initial guess of Fourier coefficients.
        p0 = np.ones(order * 2 + 1)
        date_period = (date % self.period) / self.period
        p1, success = leastsq(self.residuals, p0,
            args=(date_period, mag, order))
        #fitted_y = self.FourierSeries(p1, date_period, order)

        #print p1, self.mean, self.median
        #plt.plot(date_period, self.mag, 'b+')
        #plt.show()

        # Derive Fourier features for the first period.
        #Petersen, J. O., 1986, A&A
        self.amplitude = np.sqrt(p1[1]**2 + p1[2]**2)
        self.r21 = np.sqrt(p1[3]**2 + p1[4]**2) / self.amplitude
        self.r31 = np.sqrt(p1[5]**2 + p1[6]**2) / self.amplitude
        self.f_phase = np.arctan(-p1[1] / p1[2])
        self.phi21 = np.arctan(-p1[3] / p1[4]) - 2. * self.f_phase
        self.phi31 = np.arctan(-p1[5] / p1[6]) - 3. * self.f_phase

        """
        # Derive a second period.
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

    def get_period_uncertainty(self, fx, fy, jmax, fx_width=100):
        """
        Get uncertainty of a period.

        The uncertainty is defined as the half width
        of the frequencies around the peak, that becomes lower than
        average + standard deviation of the power spectrum.

        Since we may not have fine resolution around the peak,
        we do not assume it is gaussian. So, no scaling factor
        of 2.355 (= 2 * sqrt(2 * ln2)) is applied.

        :param fx: An array of observed date, in days.
        :param fy: An array of observed magnitude.
        :param jmax: An array of observed magnitude.
        :param fx_width: Width of power spectrum to calculate uncertainty.
        :return: Period uncertainty.
        """

        # Get subset
        start_index = jmax-fx_width
        end_index = jmax+fx_width
        if start_index < 0:
            start_index = 0
        if end_index > len(fx) - 1:
            end_index = len(fx) - 1

        fx_subset = fx[start_index:end_index]
        fy_subset = fy[start_index:end_index]
        fy_mean = np.median(fy_subset)
        fy_std = np.std(fy_subset)

        # Find peak
        max_index = np.argmax(fy_subset)

        # Find list whose powers become lower than average + std.
        index = np.where(fy_subset <= fy_mean + fy_std)[0]

        # Find the edge at left and right. This is the full width.
        left_index = index[(index<max_index)]
        if len(left_index) == 0:
            left_index = 0
        else:
            left_index = left_index[-1]
        right_index = index[(index>max_index)]
        if len(right_index) == 0:
            right_index = len(fy_subset) - 1
        else:
            right_index = right_index[0]

        # We assume the half of the full width is the period uncertainty.
        half_width = (1./fx_subset[left_index]
            - 1./fx_subset[right_index]) / 2.
        period_uncertainty = half_width

        return period_uncertainty

    def residuals(self, pars, x, y, order):
        """
        Residual of FourierSeries.

        :param pars: Fourier series parameters.
        :param x: An array of date.
        :param y: An array of true values to fit.
        :param order: order of Fourier Series.
        """

        return y - self.fourier_series(pars, x, order)

    def fourier_series(self, pars, x, order):
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
        stetson_k = np.sum(np.fabs(residual)) \
            / np.sqrt(np.sum(residual * residual)) / np.sqrt(len(mag))

        return stetson_k

    def half_mag_amplitude_ratio(self, mag, avg, weight):
        """
        Return ratio of amplitude of higher and lower
        magnitudes than average, considering weights.

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
        lower_weighted_std = np.sum((lower_mag
            - avg)**2 * lower_weight) / lower_weight_sum

        # For higher (brighter) magnitude than average.
        index = np.where(mag <= avg)
        higher_weight = weight[index]
        higher_weight_sum = np.sum(higher_weight)
        higher_mag = mag[index]
        higher_weighted_std = np.sum((higher_mag
            - avg)**2 * higher_weight) / higher_weight_sum

        # Return ratio.
        return np.sqrt(lower_weighted_std / higher_weighted_std)

    def half_mag_amplitude_ratio2(self, mag, avg):
        """
        Return ratio of amplitude of higher and lower
        magnitudes than average.

        This ratio, by definition, should be higher for EB than for others.

        :param mag: an array of magnitudes.
        :param avg: an average of magnitudes.
        """

        # For lower (fainter) magnitude than average.
        index = np.where(mag > avg)
        fainter_mag = mag[index]

        lower_sum = np.sum((fainter_mag - avg)**2) / len(fainter_mag)

        # For higher (brighter) magnitude than average.
        index = np.where(mag <= avg)
        brighter_mag = mag[index]

        higher_sum = np.sum((avg - brighter_mag)**2) / len(brighter_mag)

        # Return ratio.
        return np.sqrt(lower_sum / higher_sum)

    def get_eta(self, mag, std):
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

    def get_cusum(self, mag):
        """
        Return max - min of cumulative sum.

        :param mag: An array of magnitudes.
        :return: Max - min of cumulative sum.
        """

        c = np.cumsum(mag - self.weighted_mean) / len(mag) / self.weighted_std

        return np.max(c) - np.min(c)

    def get_features2(self):
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
                    or name == 'n_threads' or name == 'min_period'):
                # Filter some other unnecessary features.
                if not (name == 'f' or name == 'f_phase'
                        or name == 'period_log10FAP'
                        or name == 'weight' or name == 'weighted_sum'
                        or name == 'median' or name == 'mean' or name == 'std'):
                    feature_names.append(name)

        # Sort by the names.
        # Sorting should be done to keep maintaining the same order of features.
        feature_names.sort()

        # Get feature values.
        for name in feature_names:
            feature_values.append(all_vars[name])

        return feature_names, feature_values

    def get_features(self):
        """
        Return all features with its names, regardless of being used for train and prediction.
        Sorted by the names.

        :return: Features dictionary
        """

        '''
        features = {}

        # Get all the names of features.
        all_vars = vars(self)
        for name in all_vars.keys():
            if name in feature_names_list:
                features[name] = all_vars[name]

        # Sort by the keys (i.e. feature names).
        features = OrderedDict(sorted(features.items(), key=lambda t: t[0]))

        return features
        '''

        return self.get_features_all()

    def get_features_all(self):
        """
        Return all features with its names, regardless of being used for train and prediction.
        Sorted by the names.

        :return: Features dictionary
        """

        features = {}

        # Get all the names of features.
        all_vars = vars(self)
        for name in all_vars.keys():
            if name in feature_names_list_all:
                features[name] = all_vars[name]

        # Sort by the keys (i.e. feature names).
        features = OrderedDict(sorted(features.items(), key=lambda t: t[0]))

        return features