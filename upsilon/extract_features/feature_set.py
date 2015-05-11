__author__ = 'kim'


def get_feature_set():
    """
    Return a list of features' names to be used to predict a class,
    sorted by the names.

    :return: A list of features
    """

    features = ['amplitude', 'hl_amp_ratio', 'kurtosis', 'period',
        'period_SNR', 'period_uncertainty', 'phase_cusum', 'phase_eta',
        'phi21', 'phi31', 'quartile31', 'r21', 'r31', 'shapiro_w',
        'skewness', 'slope_per10', 'slope_per90', 'stetson_k']
    features.sort()

    return features

if __name__ == '__main__':
    print get_feature_set()