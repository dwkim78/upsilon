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


## Dependency
[Python 2.7+](https://www.python.org/)

[Numpy 1.8+](http://www.numpy.org/)
  
[Scipy 0.14+](http://scipy.org/)

[Matplotlib 1.3+](http://matplotlib.sourceforge.net/)

[sklearn 0.14+](http://scikit-learn.org/stable/)

[pyFFTW 0.9.2+] (http://hgomersall.github.io/pyFFTW/)


## Installation


## Usage

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
these variables must be removed. One might want to perform
sigma clipping by magnitudes and/or errors as well.

### Extracting Features

Once you have these three variables, you can extract features as:

```python
import upsilon
e_features = upsilon.ExtractFeatures(date, mag, mag_error)
e_features.run()
features, values = e_features.run()
```

If there are no magnitude errors, you can call UPSILoN as:
```python
e_features = upsilon.ExtractFeatures(date, mag)
``` 
In this case, UPSILoN will use a standard deviation of magnitudes as errors.

UPSILoN uses multiple cores to extract features. By default, UPSILoN
uses 4 cores. If you want to increase it, do as follows:

```python
e_features = upsilon.ExtractFeatures(date, mag, mag_error, n_threads=16)
```

Using multiple cores significantly improves extracting speed.

### Classification

Before classification, one must read a Random Forest classification model as
```python
rf_model = upsilon.load_model()
```
NOTE: loading module takes ~50 seconds. Thus You must load the model only once
during the whole classification processes. 
Do <font color="red">not</font> load it multiple times.

Now you can classify the light curve as

```python
class, probability = rf_model.predict(features, values)
```

That's all! Now you know ```class``` of your light curve,
and its ```probability``` as well.

### ChangeLog

#### v1.0 (planned)
- release of the first version of UPSILoN.

#### v0.5 (planned)
- add (a) Random Forest classification models.

#### v 0.2.2
- add a module estimating period uncertainty.

#### v0.2.1
- add the UPSILoN logo image.

#### v0.2
- an improved period extracting module using pyFFTW and multi-threads, 
which decreases the extracting time by ~40%.

#### v0.1
- add feature extracting modules.