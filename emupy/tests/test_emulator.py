import numpy as np
import warnings
from scipy import stats, linalg
from emupy import Emulator

def gen_data():
    X = np.vstack([stats.norm.rvs(1, 3, 500), stats.norm.rvs(-5, 2, 500)]).T
    y = stats.multivariate_normal.rvs(np.ones(3)*100, np.eye(3)*5, 500)
    return X, y


def test_sphere():
    np.random.seed(0)
    X, y = gen_data()
    E = Emulator()

    E.sphere(X, save=True)

    # check data features are whitened and centered
    assert np.isclose(np.std(E.Xsph, axis=0), 1, atol=0.05).all()
    assert np.isclose(np.median(E.Xsph, axis=0), 0, atol=0.05).all()

    # check unsphere
    X_unsphere = E.unsphere(E.Xsph)
    assert np.isclose(X_unsphere, X).all()

def test_scale_data():
    np.random.seed(0)
    X, y = gen_data()
    E = Emulator()

    for lognorm in [False, True]:
        E.scale_data(y, center=True, whiten=True, save=True, lognorm=lognorm)

        # check data features are whitened and centered
        assert np.isclose(np.std(E.y_scaled, axis=0), 1, atol=0.05).all()
        assert np.isclose(np.median(E.y_scaled, axis=0), 0, atol=0.05).all()

        # check unsphere
        y_unscaled = E.unscale_data(E.y_scaled)
        assert np.isclose(y_unscaled, y).all()

def test_klt():
    np.random.seed(0)
    X, y = gen_data()
    E = Emulator()

    # test full KLT
    E.klt(y, normalize=True, N_modes=y.shape[1])

    # check lossless transformation with all modes
    assert np.isclose(E.eig_vals.sum() / E._eig_vals.sum(), 1)
    assert np.isclose(E.klt_reproject(E.w), y).all()
    assert np.isclose(E.klt_project(y), E.w).all()
    assert np.isclose(np.std(E.w, axis=0), 1, atol=0.05).all()

    # test error propagation from std_w -> std_y via monte carlo
    std = np.ones(3) * 10
    _, std_reproj = E.klt_reproject(E.w, error=std[None, :])  # analytic reprojected std
    err_trials = stats.multivariate_normal.rvs(np.zeros(3), np.diag(std)**2, 10000)
    err_reproj = E.klt_reproject(err_trials)  # propagated MC trials
    std_err_reproj = np.std(err_reproj, axis=0)  # std of propagated MC trials
    assert np.isclose((std_reproj - std_err_reproj) / std_reproj, 0, atol=0.05).all()

    # test partial KLT
    E.klt(y, normalize=True, N_modes=2)
    assert E.eig_vals.sum() < E._eig_vals.sum()
    assert E.w.shape[1] == 2
    assert np.isclose(E.klt_project(y), E.w).all()


def test_tree():
    np.random.seed(0)
    X, y = gen_data()
    E = Emulator()
    E.create_tree(X)
    # test brute-force and tree search
    x_d_tree, x_NN_tree = E.nearest_X(X[:1], k=10, use_tree=True)
    x_d_bf, x_NN_bf = E.nearest_X(X[:1], k=10, X=X)
    assert np.isclose(x_d_tree, x_d_bf).all()
    assert np.isclose(x_NN_tree, x_NN_bf).all()



