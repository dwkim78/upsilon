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

from upsilon.test.extract_features import run as test_extract_feature
