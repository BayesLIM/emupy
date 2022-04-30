"""
Linear Model Emulator
"""
import numpy as np
import numexpr
from sklearn import linear_model
import warnings
from collections.abc import Iterable
import itertools
import copy

try:
    import sympy
    sympy_import = True
except:
    warnings.warn("Could not import sympy")
    sympy_import = False

from .emulator import Emulator
from . import utils


def get_poly_terms(Nfeatures, degree, feature_degree=False):
    """
    Function for getting power of polynomial terms
    given Nfeatures with largest power of degree

    Parameters
    ----------
    Nfeatures : int
        Number of dependent axes (or features), e.g. Nfeatures=2 for x, y

    degree : int or list
        Degree of largest polynomial. If feature_degree == False (default),
        this caps the largest polynomial term at degree.
        e.g. degree=2 yields 1,x,x^2,y,y^2,xy (note no cross terms like x^2y or x^2y^2)
        If feature_degree == True, this sets the maximum degree for each feature.
        e.g. degree=2 for 1,x,x^2,y,y^2,xy,x^2y,xy^2,x^2y^2

    feature_degree : bool, optional
        Determines whether degree sets a cap on the largest polynomial degree,
        or the largest feature degree. If True, can pass degree as a list of int
        specifying the degree of each feature.

    Returns
    -------
    list of tuples
        List of polynomial terms, with tuples indicating
        the power of each variable in the term
    """
    # setup degree for each feature, and get largest degree
    if isinstance(degree, (list, tuple, np.ndarray)):
        assert len(degree) == Nfeatures
        assert feature_degree
        max_degree = max(degree)
    else:
        max_degree = degree
        degree = [max_degree for i in range(Nfeatures)]

    # setup all feature permutations up to max_degree order
    terms = np.array(list(itertools.product(range(max_degree + 1), repeat=Nfeatures)))

    if not feature_degree:
        # if not feature_degree, eliminate excess terms
        terms = terms[terms.sum(axis=1) <= max_degree]
    else:
        # eliminate excess terms for each feature if feature_degree
        for i in range(Nfeatures):
            keep = terms[:, i] <= degree[i]
            terms = terms[keep]
    
    return terms


def get_poly_expr(poly_terms):
    """
    Given output from get_poly_terms, turn into a polynomial expression.

    Parameters
    ----------
    poly_terms : list of tuples
        Each element being a term, each integer being the power of
        the variable in the polynomial term

    Returns
    -------
    list of str
        Polynomial expressions. For each variable, x_N_M,
        N indexes a unique feature (aka variable), while
        M denotes the power of that variable in the term.
        E.g. x_0_0, x_0_1, x_1_2 = x0^0, x0^1, x1^2
    """
    terms = []
    for term in poly_terms:
        expr = ''
        for i, coeff in enumerate(term):
            expr += "x_{}_{}*".format(i, coeff)
        terms.append(expr[:-1])
    return terms


def set_poly_basis(term_expr, basis):
    """
    Set the polynomial basis

    Parameters
    ----------
    term_expr : str or list of str
        polynomial term expression (e.g. x_0_0 * x_1_1)
        output from get_poly_expr()

    basis : str
        Polynomial basis to use.
        ['direct', 'legendre', 'chebyshevt', 'chebyshevu']
        Direct is a standard polynomial (e.g. x_0_1 = x_0**1)

    Returns
    -------
    str
        New expression for the polynomial term in basis
    """
    if basis != 'direct':
        assert sympy_import, "Need sympy for basis != 'direct'"

    # check for iterable
    if isinstance(term_expr, Iterable) and not isinstance(term_expr, str):
        return [transform_poly_basis(t, basis) for t in poly_expr]

    # split term into individual variables
    variables = term_expr.split('*')

    # iterate over variables
    new_expr = []
    for v in variables:
        # get degree of this variable
        v_split = v.split('_')
        variable, deg = '_'.join(v_split[:2]), int(v_split[-1])

        # if direct basis, then evaluate exponential
        if basis == 'direct':
            new_v = "{}**{}".format(variable, deg)
        # otherwise evaluate alternative polynomial basis of degree "deg"
        elif basis == 'legendre':
            new_v = sympy.legendre_poly(deg, x=sympy.symbols(variable))
        elif basis == 'laguerre':
            new_v = sympy.laguerre_poly(deg, x=sympy.symbols(variable))
        elif basis == 'chebyshevt':
            new_v = sympy.chebyshevt_poly(deg, x=sympy.symbols(variable))
        elif basis == 'chebyshevu':
            new_v = sympy.chebyshevu_poly(deg, x=sympy.symbols(variable))
        else:
            raise ValueError("{} basis unknown".format(basis))

        new_expr.append("({})".format(new_v))

    new_expr = '*'.join(new_expr)

    return new_expr


def setup_polynomial(X, degree, basis='direct'):
    """
    Setup polynomial A matrix

    Parameters
    ----------
    X : array_like, (Nsamples, Nfeatures)
        Feature values for training data

    degree : int
        Maximum polynomial degree

    basis : str
        Polynomial basis to use.
        ['direct', 'legendre', 'chebyshevt', 'chebyshevu']
        Direct is a standard polynomial (e.g. x_0_1 = x_0**1)

    Returns
    -------
    array_like
        polynomial A matrix of shape (Nsamples, Ncoeffs)

    str
        Polynomial string mapping rows in A to rows in X
    """
    Nsamples, Nfeatures = X.shape

    # get polynomial terms
    poly_terms = get_poly_terms(Nfeatures, degree)
    Nterms = len(poly_terms)
    poly_expr = get_poly_expr(poly_terms)

    # set polynomial basis
    poly_expr = [set_poly_basis(expr, basis) for expr in poly_expr]

    # form dictionary mapping variables to X features
    Xfeat_to_terms = {"x_{}".format(i): X[:, i] for i in range(Nfeatures)}

    # iterate over terms and evaluate
    A = np.ones((Nsamples, Nterms), dtype=np.float)
    for i in range(Nterms):
        A[:, i] = numexpr.evaluate(poly_expr[i], local_dict=Xfeat_to_terms)

    return A, poly_expr


class LinearEmulator(Emulator):
    """A subclass of the Emulator class for
    linear model based emulation
    """
    def __init__(self):
        """
        A linear model based emulator, y = A c,
        where y are training data targets, A is the design
        matrix of an adopted linear model, and c are its
        coefficients.
        """
        super(LinearEmulator, self).__init__()

    def train(self, X, y, degree, basis='direct', regressor=None, modegroups=None, pool=None, **kwargs):
        """
        Train linear model on training set y at feature values X

        Sets self.models, self.A, self.modegroups, self.Ntargets

        Parameters
        ----------
        X : array_like, (Nsamples, Nfeatures)
            Feature values for training data

        y : array_like, (Nsamples, Ntargets)
            Targets for training data

        regressor : sklearn.linear_model.LinearRegression object or equivalent
            Regression model to use, see sklearn.linear_model for alternatives

        modegroups : list of lists
            List of y target indices to train and predict together
            with the same regressor. Default is each modegroup gets
            its own regressor.

        pool : multiprocess.Pool object
            Pool object for GP training, parallelized over modegroups

        kwargs : dict
            Keyword arguments to pass to regressor.fit(...)
        """
        # checks
        if regressor is None:
            regressor = linear_model.LinearRegression(fit_intercept=False)
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

        # setup the polynomial model
        self.degree, self.basis = degree, basis
        A, _ = setup_polynomial(X, degree, basis=basis)

        # setup models
        self.models = [copy.deepcopy(regressor) for i in range(self.Nmodegroups)]
        def fit(m, A=A, y=y, kwargs=kwargs):
            model, modegroup = m
            model.fit(A, y[:, modegroup], **kwargs)
            return 0

        out = list(M(fit, zip(self.models, self.modegroups)))

    def predict(self, X, unscale=False, reproject=False, **kwargs):
        """
        Predict using most recently trained models.

        Parameters
        ----------
        X : array_like, (Nsamples, Nfeatures)
            Feature values for prediction
        """
        # get A matrix
        A, _ = setup_polynomial(X, self.degree, self.basis)

        # get prediction
        pred = np.hstack([model.predict(A, **kwargs) for model in self.models])

        # reorganize according to modegroups
        sort = np.argsort(utils.flatten(self.modegroups))
        pred = pred[:, sort]

        # unscale data if scaled
        if unscale:
            pred = self.unscale_data(pred)

        # reproject if KL basis
        if reproject:
            pred = self.klt_reproject(pred)

        return pred

