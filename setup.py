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
    url             = "http://github.com/nkern/emupy",
    packages        = ['emupy','emupy.scripts'],
    setup_requires  = ['pytest-runner'],
    tests_require   = ['pytest']
    )


