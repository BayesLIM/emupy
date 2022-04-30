import sys
import os
try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name            = 'emupy',
    version         = '1.0.0',
    description     = 'emulators in python',
    author          = 'Nick Kern',
    url             = "https://github.com/BayesLIM/emupy",
    packages        = ['emupy','emupy.data'],
    tests_require   = ['pytest']
    )


