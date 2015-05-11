from setuptools import find_packages, setup

setup(
    name='upsilon',
    version='1.0',
    description='Automated Classification of Periodic Variable Stars Using Machine Learning',
    platforms=["any"],
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/dwkim78/upsilon',
    license='MIT',
    author='Dae-Won Kim',
    author_email='dwkim78@gmail.com',
    install_requires=['matplotlib>=1.4.3', 'numpy>=1.9.2',
        'scikit-learn>=0.16.1', 'scipy>=0.15.1',
        #'pyfftw>=0.9.2',
        ]
)
