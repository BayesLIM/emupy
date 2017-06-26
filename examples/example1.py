"""
example1.py
-----------

A simple demonstration of how to use emupy

"""
# Import Modules
import numpy as np
import emupy
import cPickle as pkl
import sklearn.gaussian_process as gp
import astropy.stats as astats

# Load some training data
with open('training_data.pkl','rb') as f:
    inp = pkl.Unpickler(f)
    dic = inp.load()
    data = dic['data']
    grid = dic['grid']

"""
data is training data with 250 samples
each sample has 78 data elements

grid is training parameter vectors of 250 samples
in a 3 dimensional parameter space
"""

# Separate into cross validation and training set
rng = np.random.RandomState(1)
rando = rng.choice(np.arange(250), replace=False, size=200)
select = np.array(np.zeros(250), dtype=bool)
select[rando] = True
data_tr = data[select]
grid_tr = grid[select]
data_cv = data[~select]
grid_cv = grid[~select]

# Initialize Emulator and variables
E = emupy.Emu()
E.N_modes = 15       # Save 10 out of 91 eigenmodes
E.N_data = 78       # data elements
E.N_samples = 200   # training set samples 
E.N_params = 3     # dimensional parameter space
E.lognorm = True    # log normalize data
E.cov_whiten = True
E.use_pca = True

# Calculate fiducial data
fid_data = np.array(map(np.median, data_tr.T))
fid_grid = np.array(map(np.median, grid_tr.T))

# Sphere training data
E.sphere(grid_tr, fid_grid=fid_grid, save_chol=True, norotate=True)

# Perform KLT to calculate eigenmodes
E.klt(data_tr, fid_data=fid_data, normalize=True)

# Setup Gaussian Process kwargs
kernel = gp.kernels.RBF(length_scale=np.ones(E.N_params)) + gp.kernels.WhiteKernel(noise_level=1e-6)
n_restarts_optimizer = 10
optimizer='fmin_l_bfgs_b'
gp_kwargs = {'kernel':kernel, 'n_restarts_optimizer':n_restarts_optimizer, 'optimizer':optimizer}
E.gp_kwargs = gp_kwargs

# Train Emulator
E.train(data_tr, grid_tr, fid_data=fid_data, fid_grid=fid_grid, verbose=True, invL=E.invL)

# Cross Validate
E.predict(grid_cv)
frac_err = (E.recon-data_cv)/data_cv
avg_frac_err = np.array(map(astats.biweight_midvariance, frac_err.T))
