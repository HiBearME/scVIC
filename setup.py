#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()

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
    name='scvic',
    packages=find_packages(),
    url='https://github.com/HibearME/scVIC',
    version='1.0.1'
)
