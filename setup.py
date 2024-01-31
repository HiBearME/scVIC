#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    "scanpy<=1.7.2",
    "numpy<=1.17.0",
    "torch<=2.0",
    "matplotlib<=3.4",
    "scikit-learn<=0.22.2",
    "h5py<=2.10.0",
    "pandas<=1.0",
    "loompy<=3.0.6",
    "tqdm<=4.31.1",
    "xlrd<=1.2.0",
    "hyperopt==0.1.2",
    "anndata<=0.7.4",
    "statsmodels",
    'dataclasses; python_version > "3.7"',  # for `dataclass`
    "scikit-misc",
    "seaborn<=0.11.2",
    "numba<=0.51.2",
    "importlib-metadata<=4.8.1 "
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author='Jiankang Xiong',
    author_email='hibearme@163.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    description="scVIC: Deep generative modeling "
                "of heterogeneity for scRNA-seq data",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='scvic',
    name='scvic',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/HibearME/scVIC',
    version='1.0',
    zip_safe=False,
)