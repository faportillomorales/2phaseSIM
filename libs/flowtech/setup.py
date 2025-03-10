#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Set 09 20:23:09 2023

@author: LEMI Laboratory
"""

import os
from setuptools import setup, find_packages


with open('README.md') as f:
    README = f.read()
    
requirements = os.path.dirname(os.path.realpath(__file__))+'/requirements.txt'

if os.path.isfile(requirements):
    with open(requirements) as f:
        install_requires = f.read().splitlines()

setup(
    author="LEMI Laboratory",
    author_email="lemilaboratory@usp.br",
    name='flowtechlib',
    description='Em construcao...',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'flowtechlib': ['Data/*.hdf5'],
    },
    install_requires=install_requires
)
