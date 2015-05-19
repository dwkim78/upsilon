__author__ = 'kim'

try:
    import pyfftw

    print '-------------------'
    print '| pyFFTW detected |'
    print '-------------------'
except:
    print '-------------------------------'
    print '* WARNING: No pyFFTW detected *'
    print '-------------------------------'

from upsilon.utils import utils
from upsilon.utils.logger import Logger

from upsilon.extract_features.extract_features import ExtractFeatures
from upsilon.extract_features.is_period_alias import is_period_alias as IsPeriodAlias
from upsilon.extract_features.feature_set import get_feature_set

from upsilon.datasets.base import load_rf_model
from upsilon.predict.predict import predict

from upsilon.test.predict import run as test_predict
