"""
Base Emulator class
"""

import numpy as np
from numpy import linalg


class Emulator:
    """
    The core Emulator API. The main function of this base class
    is to provide data storage, manipulation and decomposition methods.
    
    Specific subclasses should provide their own self.train(X, y, ...) and
    self.predict(X, ...) methods.
    """
    def __init__(self):
        """
        Core emulator API for training data decomposition
        """
        pass

    def sphere(self, X, Xmean=None, L=None, save=False, cov=None, norotate=False):
        """
        Perform Cholesky decomposition to sphere X onto a whitened basis.

        Sets self.Xsph, self.Xmean, self.L, self.invL if save==True.

        Parameters
        ----------
        X : array_like, (Nsamples, Nfeatures)
            Feature values for training data

        Xmean : array_like, (1, Nfeatures)
            Mean (or fiducial point) of X

        L : array_like, (Nfeatures, Nfeatures)
            Cholesky of X. If None, solved for

        save : bool, default: False
            if True, save results to self.

        cov : callable
            Covariance estimator for cholesky, default is np.cov

        norotate : bool, default: False
            if True, set off-diagonal elements of Cholesky to zero, such that
            the whitened basis is aligned with the cartesian axis.

        Returns
        -------
        array_like
            Sphered X array (Nsamples, Nfeatures)
        """
        # Get fiducial points
        if Xmean is None:
            Xmean = np.median(X, axis=0, keepdims=True)

        # Subtract mean
        Xsub = X - Xmean

        # get cholesky
        if L is None:
            # Find Covariance
            if cov is None:
                cov = np.cov
            Xcov = cov(Xsub.T)# np.cov(X.T, ddof=1) #np.inner(X.T,X.T)/self.N_samples
            if Xcov.ndim < 2:
                Xcov = np.array([[Xcov]])

            # Find cholesky
            L = linalg.cholesky(Xcov).T
            if norotate:
                L = np.eye(len(L)) * L.diagonal()
        invL = linalg.inv(L)

        # Transform to non-covarying basis
        Xsph = np.dot(invL, Xsub.T).T

        if save:
            self.Xmean, self.Xsph = Xmean, Xsph
            self.invL, self.L = invL, L

        return Xsph

    def unsphere(self, Xsph):
        """
        Unsphere Xsph given saved attributes from self.sphere()

        Parameters
        ----------
        Xsph : array_like, (Nsamples, Nfeatures)

        Returns
        -------
        array_like
            Unsphered Xsph array, (Nsamples, Nfeatures)
        """
        return Xsph @ self.L.T + self.Xmean

    def create_tree(self, X, tree_type='ball', leaf_size=100, metric='euclidean'):
        """
        Create tree structure for training features X.

        Sets self.tree object.

        Parameters
        ----------
        X : array_like, (Nsamples, Nfeatures)
            Feature values for training data

        tree_type : str [kwarg, default='ball', options=('ball','kd')]
            type of tree to make

        leaf_size : int [kwarg, default=100]
            tree leaf size, see sklearn.neighbors documentation for details

        metric : str [kwarg, default='euclidean']
            distance metric, see sklearn.neighbors documentation for details
        """
        if tree_type == 'ball':
            self.tree = neighbors.BallTree(X, leaf_size=leaf_size, metric=metric)

        elif tree_type == 'kd':
            self.tree = neighbors.KDTree(X, leaf_size=leaf_size, metric=metric)

    def nearest_X(self, x, k=10, use_tree=False, use_sph=False):
        """
        Perform a nearest neighbor search of self.X from x

        Parameters
        ----------
        x : array_like, (Nsamples, Nfeatures)
            A location(s) in self.X to perform nearest neighbor search

        k : int, default: 10
            Number of nearest neighbors to query

        use_tree : bool
            if True, use tree structure to make query, else use brute-force search

        use_sph : bool
            If True, query self.Xsph instead of self.X (only for brute-force)

        Returns
        -------
        array_like
            nearest neighbor distances

        array_like
            indices of k nearest neighbors in self.X
        """
        if use_tree:
            assert hasattr(self, 'tree'), "Must first intantiate tree"
            x_dist, x_NN = self.tree.query(x, k=k+1)
            if x.ndim == 1:
                x_dist, x_NN = x_dist[0], x_NN[0]

        else:
            if use_sph:
                assert hasattr(self, 'Xsph'), "First use sphere() for use_sph"
                X = self.Xsph
            else:
                X = self.X
            if x.ndim == 1:
                x = [x]

            # brute-force search
            dist = np.array([linalg.norm(X - _x) for _x in x])
            nearest = np.array([np.argsort(d) for d in dist])
            x_NN = np.array([n[:k+1] for n in nearest])
            x_dist = np.array([dist[n] for n in x_NN])

        if np.isclose(x_dist[0], 0):
            x_dist = x_dist[1:]
            x_NN = x_NN[1:]
        else:
            x_dist = x_dist[:-1]
            x_NN = x_NN[:-1]

        return x_dist, x_NN

    def scale_data(self, y, center=True, whiten=True, y_center=None, y_scaled_std=None, lognorm=False, save_scaling=False):
        """
        Performs a centering and (optional) rescaling of the
        training data targets.

        Sets self.y_scaled, self.y_center, self.y_scaled_std
        self.lognorm

        Parameters
        ----------
        y : array_like, (Nsamples, Ntargets)
            Training data targets

        center : bool
            If True, center y by y_center

        whiten : bool
            If True, re-scale y data to whiten the variance to unity

        y_center : array_like, (Nfeatures,)
            Center point of y. Default is median of y

        y_scaled_std : array_like, (Nfeatures,)
            Standard deviation of centered data to whiten by

        lognorm : bool
            If True, cast y into log space before center and scaling

        save_scaling : bool
            If True, save y_center and y_scaled_std to self
        """
        # Compute center
        if center:
            if y_center is None:
                if lognorm:
                    y_center = np.exp(np.median(np.log(y), axis=0))
                else:
                    y_center = np.median(y, axis=0)

        # Center data
        if lognorm:
            y_scaled = np.log(y)
            if y_center is not None:
                y_scaled -= np.log(y_center)
            self.lognorm = True
        else:
            y_scaled = y
            if y_center is not None:
                y_scaled -= y_center
            self.lognorm = False

        # whiten by MAD std
        if whiten:
            if y_scaled_std is None:
                y_scaled_std = np.median(np.abs(y_scaled), axis=0, keepdims=True) * 1.4826
            y_scaled /= y_scaled_std

        if save_scaling:
            self.y_center = y_center
            self.y_scaled = y_scaled
            self.y_scaled_std = y_scaled_std

        return y_scaled

    def unscale_data(self, y, error=None):
        """
        Given parameters set by self.scale_data, unscale y

        Parameters
        ----------
        y : array_like, (Nsamples, Nfeatures)
            Scaled y target values to unscale given saved scalings

        error : array_like, (Nsamples, Nfeatures)
            Error variance on scaled y to unscale

        Returns
        -------
        array_like
            unscaled y

        array_like
            unscaled variance on y
        """
        # unscale by std if whitened
        if hasattr(self ,'y_scaled_std') and self.y_scaled_std is not None:
            y = y * self.y_scaled_std
            if error is not None:
                error = error * self.y_scaled_std**2

        if self.lognorm:
            y = np.exp(y)
            if self.y_center is not None:
                y += np.exp(self.y_center)
            if error is not None:
                error = y**2 * error
        else:
            if self.y_center is not None:
                y = y + self.y_center

        if error is None:
            return y
        else:
            return y, error

    def klt(self, y, cov=None, normalize=True, N_modes=None):
        """
        Perform Karhunen Loeve Transform (KLT) decomposition on y.

        Sets self.eig_vals, self.eig_vecs, self.w, self.w_norm.

        Parameters
        ----------
        y : array_like, (Nsamples, Ntargets)
            Training set target values

        cov : callable
            Covariance estimator for cholesky, default is np.cov

        normalize : bool, default: False
            If True, normalize eigenvector weights to have variance of unity

        Notes
        -----
        w is y projected onto eig_vecs and divided by w_norm.
        self._eig_vals, self._eig_vecs are the un-truncated eigenvectors
        """
        if cov is None:
            cov = np.cov

        # Get covariance
        ycov = cov(y.T)

        # Solve for eigenvectors and values using SVD
        u, eig_vals, eig_vecs = linalg.svd(ycov)

        # Sort by eigenvalue
        eigen_sort = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[eigen_sort]
        eig_vecs = eig_vecs[eigen_sort]

        # Truncate eigenmodes to N_modes
        if N_modes is None:
            N_modes = len(eig_vals)

        # truncate
        self._eig_vals, self._eig_vecs = eig_vals, eig_vecs
        eig_vals = eig_vals[:N_modes]
        eig_vecs = eig_vecs[:N_modes]

        # Solve for per-sample eigenmode weight constants
        w = np.dot(y, eig_vecs.T)

        if normalize:
            self.w_norm = np.sqrt(eig_vals)
        else:
            self.w_norm = np.ones(N_modes)

        self.w = w / self.w_norm
        self.eig_vals, self.eig_vecs = eig_vals, eig_vecs

    def klt_project(self, y):
        """
        Project y target values onto previously computed eigenvectors

        Parameters
        ----------
        y : array_like, (Nsamples, Ntargets)
            Target data to project onto KLT basis
        """
        return np.dot(y, self.eig_vecs.T) / self.w_norm

    def klt_reproject(self, w, error=None):
        """
        Given parameters set by self.klt(), take weights w
        and reproject them to y space

        Parameters
        ----------
        w : array_like, (Nsamples, Nmodes)
            KLT weights

        error : array_like, (Nsamples, Nmodes)
            This is the diagonal of the covariance matrix of w.
            Reproject onto the diagonal of the cov. mat. of y.

        Returns
        -------
        array_like
            w array reprojected by KLT

        array_like
            reprojected w covariance, if error is not None
        """
        # de-project w
        y = (w * self.w_norm) @ self.eig_vecs

        if error is None:
            return y
        else:
            error = w * self.w_norm**2 @ (self.eig_vecs @ self.eig_vecs.T.conj())
            return y, error

