__author__ = 'kim'

def is_period_alias(period):
    """
    Check if a given period is possibly an alias.

    :param period: A period to test if it is a possible alias or not.
    :return: True if the given period is in a range of period alias.
    """

    # Based on the period vs periodSN plot of EROS-2 dataset (Kim+ 2014).
    # Period alias occurs mostly at ~1 and ~30.
    # Check each 1, 2, 3, 4, 5 factors.
    for i in range(1, 6):
        # One-day and one-month alias
        if (.99 / float(i)) < period < (1.004 / float(i)):
            return True
        if (1.03 / float(i)) < period < (1.04 / float(i)):
            return True
        if (29.2 / float(i)) < period < (29.9 / float(i)):
            return True

        # From candidates from the two fields 01, 08.
        # All of them are close to one day (or sidereal) alias.
        if (0.96465 / float(i)) < period < (0.96485 / float(i)):
            return True
        if (0.96725 / float(i)) < period < (0.96745 / float(i)):
            return True
        if (0.98190 / float(i)) < period < (0.98230 / float(i)):
            return True
        if (1.01034 / float(i)) < period < (1.01076 / float(i)):
            return True
        if (1.01568 / float(i)) < period < (1.01604 / float(i)):
            return True
        if (1.01718 / float(i)) < period < (1.01742 / float(i)):
            return True

        # From the all candidates from the entire LMC fields.
        # Some of these could be overlapped with the above cuts.
        if (0.50776 / float(i)) < period < (0.50861 / float(i)):
            return True
        if (0.96434 / float(i)) < period < (0.9652 / float(i)):
            return True
        if (0.96688 / float(i)) < period < (0.96731 / float(i)):
            return True
        if (1.0722 / float(i)) < period < (1.0729 / float(i)):
            return True
        if (27.1 / float(i)) < period < (27.5 / float(i)):
            return True

    # Not in the range of any alias.
    return False
