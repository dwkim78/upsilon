__author__ = 'kim'


def get_feature_set():
    """
    Return a list of features' names to be used to train a model and predict a class,
    sorted by the names.

    :return: A list of features
    """

    features = ['amplitude', 'hl_amp_ratio', 'kurtosis', 'period',
        'phase_cusum', 'phase_eta', 'phi21', 'phi31', 'quartile31',
        'r21', 'r31', 'shapiro_w', 'skewness', 'slope_per10',
        'slope_per90', 'stetson_k']
    features.sort()

    return features


def get_feature_set_all():
    """
    Return a list of entire features regardless of being used to train
    a model and predict a class.
    :return: A list of features.
    """

    features = get_feature_set()

    features.append('cusum')
    features.append('eta')
    features.append('n_points')
    features.append('period_SNR')
    features.append('period_log10FAP')
    features.append('period_uncertainty')
    features.append('weighted_mean')
    features.append('weighted_std')

    features.sort()

    return features


if __name__ == '__main__':
    print get_feature_set()