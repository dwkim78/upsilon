# UPSILoN [ Under Development ]
<div align="center">
<img src="https://github.com/dwkim78/upsilon/blob/master/UPSILoN.png">
</div><br>

<font size="4">UPSILoN (<font color="red"><b>U</b></font>biquitous Classification 
of <font color="red"><b>P</b></font>eriodic Variable <font color="red"><b>S</b></font>tars 
using Mach<font color="red"><b>I</b></font>ne 
<font color="red"><b>L</b></font>ear<font color="red"><b>N</b></font>ing) 
aims to classify periodic variable stars 
(e.g. *[Delta Scuti](http://en.wikipedia.org/wiki/Delta_Scuti_variable),
[RR Lyraes](http://en.wikipedia.org/wiki/RR_Lyrae_variable),
[Cepheids](http://en.wikipedia.org/wiki/Cepheid_variable),
[eclipsing binaries](http://en.wikipedia.org/wiki/Binary_star#Eclipsing_binaries),
and [long-period variables](http://en.wikipedia.org/wiki/Long-period_variable_star)*) 
using only single-band optical light-curves **regardless of** survey-specific characteristics 
such as color, magnitude, sampling rate, etc (Kim+ 2015 in preparation).</font>


## 1. Dependency
[Python 2.7+](https://www.python.org/)

[Numpy 1.8+](http://www.numpy.org/)
  
[Scipy 0.14+](http://scipy.org/)

[Matplotlib 1.3+](http://matplotlib.sourceforge.net/)

[scikit-learn 0.14+](http://scikit-learn.org/stable/)

[pyFFTW 0.9.2+] (http://hgomersall.github.io/pyFFTW/) [Optional but highly recommended for multi-threads usage]


## 2. Installation
- NOT YET

## 3. Pseudo Code: How to Use UPSILoN? 

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
    mag_error = np.array([...])
    
    # Pre-process and/or refine the light curve.
    == Do it yourself ==
    
    # Extract features
    e_features = upsilon.ExtractFeatures(date, mag, mag_error)
    e_features.run()
    features, values = e_features.get_features()
    
    # Classify the light curve
    label, probability = rf_model.predict(features, values)
    print label, probability
```


## 4. Details about UPSILoN Usage

### Preparation

In order to extract features from a light curve, one must prepare
three variables, each of which is an numpy array of dates in days, 
magnitudes, and magnitude errors. For example, see the following pseudo code.

```python
date = np.array([...])
mag = np.array([...])
mag_error = np.array([...])
```

Note: An array of magnitude errors is not mandatory. 

If necessary, these three variables must be refined prior to be ingested
to UPSILoN. For instance, invalid values (e.g. nan, inf, etc.) among 
these variables must be removed. UPSILoN only provides a sigma-clipping routine.
You can use it as:

```python
date, mag, mag_error = upsilon.utils.sigma_clipping(date, mag, mag_error
    threshold=3, iteration=1)
```

This removes fluctuated data points in magnitudes.
By default, UPSILoN remove data points with 3 sigma-threshold 
with one iteration.

### Extracting Features

Once you have these three variables, you can extract features as:

```python
import upsilon
e_features = upsilon.ExtractFeatures(date, mag, mag_error)
e_features.run()
features, values = e_features.get_features()
```

If there are no magnitude errors, you can call UPSILoN as:
```python
e_features = upsilon.ExtractFeatures(date, mag)
``` 
In this case, UPSILoN will use a standard deviation of magnitudes as errors.

If pyFFTW is installed, UPSILoN can use multiple cores when estimating periods.
Note that estimating periods takes at least a few thousands times more than
estimating all other features (e.g. 2-4 seconds versus 0.001 seconds)
without using multiple cores.
By default, UPSILoN uses 4 cores. If you want to use more, do as follows:

```python
e_features = upsilon.ExtractFeatures(date, mag, mag_error, n_threads=8)
```

Using Macbook Air 2012 13-inch model equipped with Intel Core i5 1.8 GHz 
(2 cores and total 4 threads) and 8 GBytes memory,
extracting a period takes 1-2 second per light curve
containing ~300-1000 data points.

### Classification

Before predicting a class, 
one must read a Random Forest classification model as
```python
rf_model = upsilon.load_rf_model()
```
NOTE: Loading a model takes ~30 seconds. Thus you must 
<font color="red"><b>NOT</b></font> load it multiple times.

Now you can classify the light curve as

```python
label, probability = rf_model.predict(features, values)
```

That's all! Now you know the class of your light curve, ```label```, 
and its class probability, ```probability```, as well.

### Tip

By the nature of UPSILoN, it can distinguish 
periodic variable light curves from non-varying light curves (i.e. non-variables).
Nevertheless, since feature extracting takes lots of time,
removal of non-varying light curves prior to running UPSILoN 
would significantly reduces the total processing time.
Unfortunately, there is no universal way to remove such non-varying light curves
for many different time-series surveys, and thus UPSILoN do not provides such functionality.
Each UPSILoN user has their own choices of what to remove.


## 5. UPSILoN Classification Performance
 
### Assessment of Classification Model

### Application to Astronomical Survey

#### EROS-2

#### MACHO

#### ASAS

## 6. ChangeLog

### v1.0 (planned)
- release of the first version of UPSILoN.

### v0.5 (planned)
- add (a) Random Forest classification models.

### v0.2.3
- add a module calculating a feature based on cumulative sum.

### v0.2.2
- add a module estimating period uncertainty.

### v0.2.1
- add the UPSILoN logo image.

### v0.2
- an improved period extracting module using pyFFTW and multi-threads, 
which decreases the extracting time by ~40%.

### v0.1
- add feature extracting modules.