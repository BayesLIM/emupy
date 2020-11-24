import numpy as np
from emupy import LinearEmulator
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

def test_linear():
    # test easy example of emulating and reproducing 1D data
    (data_cr1, grid_cr1, data_cr2, grid_cr2,
     data_cr3, grid_cr3) = load_data()

    X, y = grid_cr1[:, :1], data_cr1[:, :1]

    E = LinearEmulator()

    E.scale_data(y, center=True, lognorm=True, save=True)
    E.train(X, E.y_scaled, 4, basis='direct')
    pred = E.predict(X, unscale=True)

    # assert prediction is a decent match (to below 10%)
    assert np.isclose(np.mean(abs(pred - y)/pred), 0, atol=0.1).all()

    # try with different poly basis
    E.train(X, E.y_scaled, 4, basis='legendre')
    pred2 = E.predict(X, unscale=True)
    assert np.isclose(np.mean(abs(pred2 - y)/pred2), 0, atol=0.1).all()
