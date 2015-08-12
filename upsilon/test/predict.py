__author__ = 'kim'

import time
from upsilon.extract_features.extract_features import ExtractFeatures
from upsilon.datasets.base import load_EROS_lc
from upsilon.datasets.base import load_rf_model
from upsilon.predict.predict import predict

from upsilon.extract_features.feature_set import get_feature_set

from upsilon.utils.utils import sigma_clipping
from upsilon.utils.logger import Logger

feature_names_list = get_feature_set()

def run():
    """
    Test UPSILoN package.

    :return: None
    """

    logger = Logger().getLogger()

    logger.info('Read a light curve')
    date, mag, err = load_EROS_lc()

    index = mag < 99.999
    date = date[index]
    mag = mag[index]
    err = err[index]

    logger.info('   Before sigma-clipping: %d data points' % len(date))
    #date, mag, err = sigma_clipping_without_error(date, mag)
    date, mag, err = sigma_clipping(date, mag, err)
    logger.info('   After sigma-clipping: %d data points' % len(date))

    for i in range(1):
        start = time.time()
        e_features = ExtractFeatures(date, mag, err)
        e_features.run()
        end = time.time()
        logger.info('Feature extracting time: %.4f seconds' % (end - start))

    logger.info('Extracted features. %d of these, marked with (+) '
        'are used to predict a class:'
        % len(feature_names_list))

    features_all = e_features.get_features()
    for key, value in features_all.iteritems():
        if key != 'n_points':
            logger.info('   %s %s: %f' % ('(+)'
                if key in feature_names_list else '(-)', key, value))
        else:
            logger.info('   %s %s: %d' % ('(+)'
                if key in feature_names_list else '(-)', key, value))

    logger.info('Load the UPSILoN classifier')
    rf_model = load_rf_model()

    label, prob, flag = predict(rf_model, features_all)
    logger.info('Classify the light curve')
    logger.info('   Classified as %s with the class probability %.2f' %
        (label, prob))
    logger.info('   Classification flag: %d', flag)

    logger.handlers = []

if __name__ == '__main__':
    run()