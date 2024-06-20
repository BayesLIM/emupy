## emupy : Emulators in Python
![Run Tests](https://github.com/BayesLIM/emupy/workflows/Run%20Tests/badge.svg)
[![codecov](https://codecov.io/gh/BayesLIM/emupy/branch/main/graph/badge.svg?token=3Q1IZUGZ5W)](https://codecov.io/gh/BayesLIM/emupy)


### Version: 1.0.0
Code Repo : https://github.com/BayesLIM/emupy

### About:
*emupy* is a generalized Python implementation of the emulator algorithm detailed in [Kern et al. 2017](https://arxiv.org/abs/1705.04688),
and includes methods for Principcal Component Analysis and Gaussian Process Regression. To see emupy applied specifically to emulating the 21cm EoR power spectrum, see [pycape](https://github.com/nkern/pycape).

*emupy* version 1.0.0 has an updated API, and now also includes a generalized linear model emulator (`emupy.linear.LinearEmulator`) and a neural network emulator built on pytorch (`emupy.nn.NNEmulator`).
emupy is only compatible with Python >= 3.6.


### Dependencies:
- numpy >= 1.10.4
- scipy >= 1.4.0
- sklearn >= 0.18

#### Optional Dependencies:
- pytorch >= 1.7.0
- sympy >= 1.3
- numexpr >= 2.6.9

### Install:
To install, clone the repo, cd into it and run the setup.py script
```bash
python setup.py install
```
or
```bash
pip install -e .
```

### Running:
See Examples for demonstrations on how to run the code

### Citation:
Please use [Kern et al. 2017](https://arxiv.org/abs/1705.04688) for citation.

### Authors:
Nicholas Kern, UC Berkeley
<br>
Duncan Rocha, Harvey Mudd College
