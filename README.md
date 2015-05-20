# UPSILoN
<div align="center">
<img src="https://github.com/dwkim78/upsilon/blob/master/upsilon/datasets/images/UPSILoN.png">
</div><br>

UPSILoN (A<b>U</b>tomated Classification of <b>P</b>eriodic Variable <b>S</b>tars 
using Mach<b>I</b>ne <b>L</b>ear<b>N</b>ing) aims to classify periodic variable stars 
such as [Delta Scuti stars](http://en.wikipedia.org/wiki/Delta_Scuti_variable),
[RR Lyraes](http://en.wikipedia.org/wiki/RR_Lyrae_variable),
[Cepheids](http://en.wikipedia.org/wiki/Cepheid_variable),
[Type II Cepheids] (http://en.wikipedia.org/wiki/Type_II_Cepheid),
[eclipsing binaries](http://en.wikipedia.org/wiki/Binary_star#Eclipsing_binaries), and
[long-period variables](http://en.wikipedia.org/wiki/Long-period_variable_star) 
(i.e. superclasses), and their subclasses (e.g. RR Lyrae ab, c, d, and e types) 
using well-sampled optical-band light curves **regardless of** survey-specific characteristics 
such as color, magnitude, sampling rate, etc (Kim+ 2015 in preparation).

Note: In principle, UPSILoN can classify 
any light curves having arbitrary number of data points. 
However, for the best classification quality,
we recommend to use light curves with more than ~100 data points,
and have been observed longer than three months.


## 1. Dependency
[Python 2.7+](https://www.python.org/) 

 * Not tested with Python 3.0+

[Numpy 1.9+](http://www.numpy.org/)
  
[Scipy 0.15+](http://scipy.org/)

[scikit-learn 0.16.1+](http://scikit-learn.org/stable/)

[pyFFTW 0.9.2+] (http://hgomersall.github.io/pyFFTW/) 

 * pyFFTW is optional but <b>highly recommended</b>
  for multi-threads usage for FFT.


## 2. Installation

[WARNING: UPSILoN is under the beta-test phase now.
Thus be careful to use it and if you find any bug, please contact with me.
In addition ```pip``` installation is not supported yet.]

The easiest way to install the UPSILoN package is:

```python
pip install upsilon
```

Alternatively, you can download all the necessary files from the git repository as:

```python
git clone https://github.com/dwkim78/upsilon

cd upsilon
python setup.py install
```

## 3. Test

To check if the UPSILoN library is successfully installed, type the 
following code:

```python
import upsilon
upsilon.test_predict()
```

This code reads a sample light curve of a Cepheid variable, 
extracts features, and classify it.
At the end of log messages, you must see the line something like:

```
yyyy/mm/dd hh:mm:ss INFO - Classified as CEPH_1O with the class probability 0.90
```

If the light curve is not classified as a Cepheid variable,
or if the class probability is not 0.90, something might be wrong.
In that case, please contact with me.

## 4. Pseudo Code: How to Use UPSILoN? 

The following pseudo code shows the entire processes of how to use UPSILoN.

```python
import upsilon

# Load a classification model.
rf_model = upsilon.load_rf_model()

# Extract features from each light curve and predict its class. 
for light_curve in set_of_light_curves:
    # Read the light curve's date (in days), magnitude, and magnitude errors.
    ...
    date = np.array([...])
    mag = np.array([...])
    err = np.array([...])
    
    # Pre-process and/or refine the light curve.
    == Do it yourself ==
    
    # Extract features
    e_features = upsilon.ExtractFeatures(date, mag, err)
    e_features.run()
    features = e_features.get_features()
    
    # Classify the light curve
    label, probability = upsilon.predict(rf_model, features)
    print label, probability
```


## 5. Details about UPSILoN Usage

### Preparation

In order to extract features from a light curve, one needs to prepare
three variables, each of which is a numpy array of dates in days, 
magnitudes, and magnitude errors. For example, see the following pseudo code.

```python
date = np.array([...])
mag = np.array([...])
err = np.array([...])
```

Note: An array of magnitude errors is not mandatory. 

If necessary, these three variables have to be refined prior to being ingested
to UPSILoN. For instance, invalid values (e.g. nan, inf, etc.) among 
these variables must be removed. 
For refining light curves, UPSILoN only provides a sigma-clipping routine.
You can use it as:

```python
date, mag, err = upsilon.utils.sigma_clipping(date, mag, err, 
    threshold=3, iteration=1)
```

This removes fluctuated data points in magnitudes.
By default, the module removes fluctuated data points 
with 3 sigma-threshold with one iteration.


### Extracting Features

Once you have these three variables, you can extract features as:

```python
e_features = upsilon.ExtractFeatures(date, mag, err)
e_features.run()
features = e_features.get_features()
```

If there are no magnitude errors, you can do this as well:
```python
e_features = upsilon.ExtractFeatures(date, mag)
``` 
In this case, UPSILoN will use a standard deviation of magnitudes as errors.

If pyFFTW is installed, UPSILoN utilizes multiple cores to derive a period
because the period estimation takes a lot longer than calculating all other features.
By default, UPSILoN uses 4 cores. If you want to use more, do as follows:

```python
e_features = upsilon.ExtractFeatures(date, mag, err, n_threads=8)
```

Using Macbook Air 2012 equipped with Intel Core i5 1.8 GHz 
(2 cores and total 4 threads) and 8 GBytes memory,
extracting a period takes 1-2 seconds per light curve 
containing ~200-500 data points.

After extracting features, you may want to check if the derived period
is in a range of period alias (e.g. one day, sidereal day, one month, etc).
UPSILoN provides a simple module to check it as:

```python
is_alias = upsilon.IsPeriodAlias(25.512)
```

If ```is_alias``` is ```True```, then the period is possibly an alias.
In such case, one must be careful to classify the light curve,
since classification could be wrong.
Note that we also included additional aliases empirically
determined based on the OGLE and EROS-2 dataset 
([Kim et al. 2014] (http://adsabs.harvard.edu/abs/2014A%26A...566A..43K)).


### Classification

Before predicting a class, 
one must read a Random Forest classification model as
```python
rf_model = upsilon.load_rf_model()
```
NOTE: Loading the model takes 5-20 seconds depending on machines. 
Thus, be careful not to load it multiple times.

You are now ready to classify the light curve.

```python
label, probability = upsilon.predict(rf_model, features)
```

That's all! Now you know the class of your light curve, ```label```, 
and its class probability, ```probability```, as well.

### Tip


By the nature of UPSILoN, it can distinguish 
periodic variable light curves from non-varying light curves (i.e. non-variables).
Nevertheless, since feature extraction takes lots of time,
removal of non-varying light curves before running UPSILoN 
would significantly reduces the total processing time.
Unfortunately, it is hard to find an universal and consistent way of 
removing such non-varying light curves from
many time-series surveys having different characteristics, 
so UPSILoN does not provides such functionality (yet).


### Logger

If you want to write log messages either to console or to disk, 
you can use the UPSILoN Logger class as:

```python
logger = upsilon.Logger().getLogger()

logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')
```

Keep in mind that you need to generate only one logger instance 
through the whole processes, but not many.
If you want to save log messages to a file, 
generate a logger instance as follows:
 
 ```python
 logger = upsilon.Logger('/PATH/TO/FILE.log').getLogger()
 ```

This will send log messages to both console and a log file.
Note that the path must be the absolute path.


## 6. UPSILoN Classification Quality

The UPSILoN classifier was trained on the OGLE 
([Udalski et al. 1997] (http://adsabs.harvard.edu/abs/1997AcA....47..319U)) 
and EROS-2 periodic variables 
([Kim et al. 2014] (http://adsabs.harvard.edu/abs/2014A%26A...566A..43K)).
The hyper parameters of the classifier were tuned
using grid-search and 10-fold cross-validation.

The classifier trained using only superclasses shows 99% recall and precision
whereas the classifier trained on subclasses shows 80% recall and precision.
The confusion in the subclass classifier was mainly caused by
misclassification within superclasses. 
The following figures show classification quality of these two classifiers.
In the figures, each cell is divided by the sum of the row where
the cell belongs to. Thus brighter the cell, higher the recall.

<div align="center">
<img src="https://github.com/dwkim78/upsilon/blob/master/upsilon/datasets/images/model_map_superclass.png">
<br>[ Map of the confusion matrix of the superclass model ]
<br>
<img src="https://github.com/dwkim78/upsilon/blob/master/upsilon/datasets/images/model_map_subclass.github.png">
<br>[ Map of the confusion matrix of the subclass model ]
</div><br>

UPSILoN provides the classifier trained using all the subclasses.
For the comprehensive experiments using 
the original and re-sampled MACHO and ASAS light curves,
see Kim et al. 2015 (in preparation).

Note that we provide the random forests model trained with 100 trees
and randomly selected 10 features whereas, in Kim et al. 2015 (in preparation),
we used 500 trees with randomly selected 12 features.
The [F1 score] (http://en.wikipedia.org/wiki/F1_score)
difference between these two models is 0.001, which is negligible. 
The reason is only because GitHub does not allow to upload 
a file larger than 100 MB.
The size of the model with 500 trees and 12 features is 260 MB.


## Minimum Requirements

Although UPSILoN could be run at any machines,
we recommend to run it at machines equipped with at least ~2 GB memory
because the uncompressed random forests model file could consume
a large amount of memory.


## ChangeLog

### v?.0 (planned)
- implementing multilayer classifiers, which will
 substantially reduce feature extracting time.

### v1.0 (soon)
- release of the first version of UPSILoN (Kim+ 2015 in preparation).

### v0.7 (ongoing)
- code improvements, bug fixes, tests, etc.

### v0.6
- add dataset for tests.
- add a module testing the classification model using the dataset.
- add a module for classifying a light curve.

### v0.5
- add a random forests classification model.

### v0.3.1
- add a module to check if a given period is an alias or not.

### v0.3
- structure of features is changed to Python OrderedDict type.
- add a sigma clipping module.
- add a Logger class.
- raise a warning if the number of measurements in a light curve is 
less than 100.
- other few improvements and bug fixes.

### v.0.2.4
- Bug fixed in the module estimating a period uncertainty. 

### v0.2.3
- add a module calculating a feature based on cumulative sum.

### v0.2.2
- add a module estimating period uncertainty.

### v0.2.1
- add the UPSILoN logo image.

### v0.2
- an improved period extracting module using pyFFTW and multi-threads, 
which substantially decreases the extracting time.

### v0.1
- add feature extracting modules.