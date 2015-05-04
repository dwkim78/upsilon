__author__ = 'kim'

try:
    import pyfftw

    print '-------------------'
    print '| pyFFTW detected |'
    print '-------------------'
except:
    print '----------------------------------------------------------'
    print '** WARNING: No pyFFTW detected. Numpy FFTW will be used **'
    print '----------------------------------------------------------'

from upsilon import utils
from .extract_features.extract_features import ExtractFeatures
