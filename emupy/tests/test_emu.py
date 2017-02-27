import numpy as np
import unittest
import warnings
import scipy.stats as stats
import scipy.linalg as la
from emupy import Emu

class TestEmu(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_no_pca(self):
        """
        Simple Emulator Test Using no PCA
        """
        N_data = 5
        X = np.linspace(1,10,N_data)
        y = 3.0*X + stats.norm.rvs(0,0.1,len(X))
        yerrs = np.ones(len(y))*0.1

        # Generate training set
        N_samples = 25
        data_tr = []
        grid_tr = []
        for theta in np.linspace(0.1,10,N_samples):
            data_tr.append(theta*X)
            grid_tr.append(np.array([theta]))

        data_tr = np.array(data_tr)
        grid_tr = np.array(grid_tr)

        # Instantiate
        N_modes = N_data
        variables = {'reg_meth':'gaussian','gp_kwargs':{},'N_modes':N_modes,'N_samples':N_samples,
                'scale_by_std':False,'scale_by_obs_errs':False,'lognorm':False}
        E = Emu(variables)

        E.sphere(grid_tr, save_chol=True)

        # Train
        E.fid_params = np.array([5.0])
        E.fid_data = E.fid_params[0]*X
        E.train(data_tr, grid_tr, fid_data=E.fid_data, fid_params=E.fid_params, use_pca=False, invL=E.invL)
        E.w_norm = np.ones(N_modes)
        E.recon_err_norm = np.ones(N_data)

        pred_kwargs = {'use_pca':False,'fast':True}
        _ = E.predict(np.array([3.0])[:,np.newaxis], **pred_kwargs)

    def test_cholesky(self):
        """
        Test to make sure cholesky decomposition works
        """
        # Generate Random sample
        d = np.vstack([stats.norm.rvs(100,10,1000), stats.norm.rvs(0.5,0.1,1000)]).T
        E = Emu({'N_samples':1000})
        E.sphere(d, save_chol=True)
        if E.Xsph.T[0].max() > 10 and E.Xsph.T[0].min() < -10:
            print("Cholesky probably not working correctly, perhaps la.cholesky(Xcov) is transposed?")
            self.fail("")

if __name__ == '__main__':
    unittest.main()
