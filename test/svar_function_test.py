import numpy as np
import scipy.io as spio

resid = spio.loadmat('/Users/fangli/PySVAR/PySVAR/data/resid.mat')['usim']
irf = spio.loadmat('/Users/fangli/PySVAR/PySVAR/data/irf.mat')['IRF']
rotation = spio.loadmat('/Users/fangli/PySVAR/PySVAR/data/rotation.mat')['Q']
chol = spio.loadmat('/Users/fangli/PySVAR/PySVAR/data/chol.mat')['cdc']

shocks = np.dot(np.linalg.inv(np.dot(chol, rotation)), resid.T)
