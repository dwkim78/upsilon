__author__ = 'kim'

from upsilon.extract_features.is_period_alias import is_period_alias

def predict(rf_model, features):
    """
    Return label and probability estimated using the UPSILoN
    random forests model and the input features.
    :param rf_model: The UPSILoN random forests model.
    :param features: A list of features estimated by UPSILoN
    :return: label (i.e. class) and class probability.
    """

    import numpy as np
    from upsilon.extract_features.feature_set import get_feature_set
    feature_set = get_feature_set()

    # Grab only necessary features.
    cols = [feature for feature in features if feature in feature_set]
    cols = sorted(cols)
    filtered_features = []
    for i in range(len(cols)):
        filtered_features.append(features[cols[i]])

    # Classify.
    classes = rf_model.classes_
    # Note that we're classifying a single source, so [0] need tobe added.
    probabilities = rf_model.predict_proba(filtered_features)[0]

    # Classification flag.
    flag = 0
    if features['period_SNR'] < 20. or is_period_alias(features['period']):
        flag = 1

    # Return class, probability, and flag.
    max_index = np.where(probabilities == np.max(probabilities))
    return classes[max_index][0], probabilities[max_index][0], flag
