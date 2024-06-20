"""
Gaussian Process Regression Emulator
"""
from sklearn.gaussian_process import GaussianProcessRegressor
import copy
import numpy as np

from .emulator import Emulator
from . import utils


class GPEmulator(Emulator):
    """A subclass of the Emulator class for
    Gaussian Process Regression based emulation.
    """
    def __init__(self):
        """
        A Gaussian Process (GP) based emulator
        """
        super(GPEmulator, self).__init__()

    def train(self, X, y, GP, modegroups=None, pool=None):
        """
        Train GP model on training set y at locations X by
        regressing for kernel hyperparameters and precomputing
        prediction matrices.

        Sets self.models, self.modegroups, self.Ntargets

        Parameters
        ----------
        X : array_like, (Nsamples, Nfeatures)
            Feature values for training data

        y : array_like, (Nsamples, Ntargets)
            Targets for training data

        GP : sklearn GaussianProcessRegressor object
            GP object with kernel defined

        modegroups : list of lists
            List of y target indices to train and predict together
            with the same kernel. Default is each target gets its own GP

        pool : multiprocess.Pool object
            Pool object for GP training, parallelized over modegroups
        """
        assert GP.kernel is not None, "Must define kernel function with GP"
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.Ntargets = y.shape[1]

        if modegroups is None:
            modegroups = [[i] for i in range(self.Ntargets)]
        self.modegroups = modegroups
        self.Nmodegroups = len(modegroups)

        if pool is None:
            M = map
        else:
            M = pool.map
            
        self.models = [copy.deepcopy(GP) for i in range(self.Nmodegroups)]
        def fit(m, X=X, y=y):
            model, modegroup = m
            model.fit(X, y[:, modegroup])
            return 0

        out = list(M(fit, zip(self.models, self.modegroups)))

    def predict(self, X, unscale=False, reproject=False, **kwargs):
        """
        Predict current GP models given location data X.

        Parameters
        ----------
        X : array_like, (Nsamples, Nfeatures)
            Feature values for prediction

        unscale : bool
            If trained on scaled y data, unscale prediction

        reproject : bool
            If trained on KL weights, reproject prediction
        """
        # get prediction
        pred = [model.predict(X, **kwargs) for model in self.models]
        err = None

        # parse if error is also computed
        if kwargs.get("return_std", False) or kwargs.get("return_cov", False):
            # split prediction from error
            pred, err = [np.atleast_2d(p[0].T).T for p in pred], [np.atleast_2d(p[1].T).T for p in pred]
            err = np.concatenate(err, axis=-1)
            if kwargs.get("return_std", False):
                # temporarily make err a variance for re-scaling and re-projection
                err = err**2

        else:
            pred = [np.atleast_2d(p.T).T for p in pred]

        # stack predictions along last axis
        pred = np.concatenate(pred, axis=-1)

        # reorganize according to modegroups
        sort = np.argsort(utils.flatten(self.modegroups))
        pred = pred[:, sort]
        if err is not None:
            # sort error and transpose to match pred ordering
            err = err[:, sort]

        # unscale data if scaled
        if unscale:
            pred = self.unscale_data(pred, err)
            if err is not None:
                pred, err = pred

        # reproject if KL basis
        if reproject:
            pred = self.klt_reproject(pred, err)
            if err is not None:
                pred, err = pred

        if kwargs.get("return_std", False):
            # transform err back to std as requested
            err = np.sqrt(err)

        if err is None:
            return pred
        else:
            return pred, err
