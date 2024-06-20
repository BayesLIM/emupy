import numpy as np
from sklearn import gaussian_process as gp
from emupy import GPEmulator
from emupy.data import DATA_PATH
import pickle

def load_data():
    with open(DATA_PATH+'/cross_inspection.pkl','rb') as f:
        dic = pickle.load(f, encoding='latin1')
        data_cr = dic['data']
        grid_cr = dic['grid']
        # Sort the cross_inspection data and split into three individual data sets
        fid_grid = np.median(grid_cr, axis=0)
        sort1 = np.where((grid_cr.T[1]==fid_grid[1])&(grid_cr.T[2]==fid_grid[2]))[0]
        sort2 = np.where((grid_cr.T[0]==fid_grid[0])&(grid_cr.T[2]==fid_grid[2]))[0]
        sort3 = np.where((grid_cr.T[0]==fid_grid[0])&(grid_cr.T[1]==fid_grid[1]))[0]
        data_cr1, grid_cr1 = data_cr[sort1], grid_cr[sort1]
        data_cr2, grid_cr2 = data_cr[sort2], grid_cr[sort2]
        data_cr3, grid_cr3 = data_cr[sort3], grid_cr[sort3]
    return data_cr1, grid_cr1, data_cr2, grid_cr2, data_cr3, grid_cr3

def test_gpr():
    # test easy example of emulating and reproducing 1D data
    (data_cr1, grid_cr1, data_cr2, grid_cr2,
     data_cr3, grid_cr3) = load_data()

    X, y = grid_cr1[:, :1], data_cr1[:, :1]

    E = GPEmulator()
    kernel = gp.kernels.RBF() + gp.kernels.WhiteKernel()
    GP = gp.GaussianProcessRegressor(kernel)

    E.scale_data(y, center=True, lognorm=True, save=True)
    E.train(X, E.y_scaled, GP)
    pred1 = E.predict(X, return_std=False, unscale=True)
    pred2, err = E.predict(X, return_std=True, unscale=True)

    assert np.isclose(abs(pred1 - pred2), 0, atol=1e-5).all()

    # assert prediction is a good match (to below 10%)
    assert np.isclose(abs(pred1 - y)/pred1, 0, atol=0.1).all()

    # assert error is reasonable
    assert np.isclose(np.mean(np.std(pred1 - y) / err), 1, rtol=2)


    # test for grouped modegroups
    _y = np.concatenate([y, y], axis=1)
    E.scale_data(_y, center=True, lognorm=True, save=True)
    E.train(X, E.y_scaled, GP, modegroups=[[0,1]])
    pred, err = E.predict(X, return_std=True, unscale=True)
