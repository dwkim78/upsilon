from setuptools import find_packages, setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='upsilon',
    version='1.2.1',
    description='Automated Classification of Periodic Variable Stars Using Machine Learning',
    long_description=readme(),
    platforms=['any'],
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/dwkim78/upsilon',
    license='MIT',
    author='Dae-Won Kim',
    author_email='dwkim78@gmail.com',
    install_requires=['numpy>=1.9', 'scikit-learn>=0.16.1', 'scipy>=0.15',
        #'pyfftw>=0.9.2',
    ],
    keywords=['astronomy', 'periodic variables', 'light curves',
        'variability features', 'time-series survey', 'machine learning',
        'classification'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Astronomy'
    ]
)
