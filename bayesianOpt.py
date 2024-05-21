import numpy as np

from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

from src.data import Dataset
from src.train import Training

parameters = dict(
    target = 'FP',
    subfrag_size = 4,
    edge_size = 1,
    out_size = 1,
    seed = 42,
    batch_size = 32,
    max_epoch = 200,
    verbose = False,
)

dataset = Dataset(
    target = parameters['target'],
    seed = parameters['seed'],
    batch_size = parameters['batch_size'],
    verbose = parameters['verbose']
)

def target_function(hidden_size, dropout, num_layers, num_timesteps, lr_init, gamma):
    new_params = parameters

    new_params['hidden_size'] = int(hidden_size)
    new_params['dropout'] = float(dropout)
    new_params['num_layers'] = int(num_layers)
    new_params['num_timesteps'] = int(num_timesteps)
    new_params['lr_init'] = float(pow(10, -1 * lr_init))
    new_params['gamma'] = float(gamma)
    
    train = Training(new_params, dataset = dataset)
    train.run()

    return -1 * train.mae

cov = matern32()
gp = GaussianProcess(cov)
acq = Acquisition(mode = 'ExpectedImprovement')

param = {
    'hidden_size' : ('int', [10, 100]),
    'dropout' : ('cont', [0.1, 0.5]),
    'num_layers' : ('int', [1, 5]),
    'num_timesteps' : ('int', [1, 5]),
    'lr_init' : ('int', [1, 5]),
    'gamma' : ('cont', [0.95, 0.999]),
}

np.random.seed(parameters['seed'])
gpgo = GPGO(gp, acq, target_function, param, n_jobs = 10)
gpgo.run(max_iter = 10)

gpgo.getResult()
    

