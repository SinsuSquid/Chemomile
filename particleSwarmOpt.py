import numpy as np
from src.data import Dataset
from src.train import Training

parameters = dict(
    target = 'ESOL',
    subfrag_size = 4,
    edge_size = 1,
    out_size = 1,
    seed = 42,
    batch_size = 128,
    max_epoch = 200,
    verbose = False,
)

# hidden_size, dropout, num_layers, num_timestep, lr_init, gamma
bounds = (np.array([ 10, 0.2, 1, 1, 1, 0.975]),  # lower bounds
          np.array([100, 0.5, 5, 5, 5, 0.999]))  # upper bounds

dataset = Dataset(
    target = parameters['target'],
    seed = parameters['seed'],
    batch_size = parameters['batch_size'],
    verbose = parameters['verbose']
)

def target_function(input_array):
    new_params = parameters
    input_array = input_array.flatten()
    

    new_params['hidden_size'] = int(input_array[0])
    new_params['dropout'] = float(input_array[1])
    new_params['num_layers'] = int(input_array[2])
    new_params['num_timesteps'] = int(input_array[3])
    new_params['lr_init'] = float(pow(10, -1 * input_array[4]))
    new_params['gamma'] = float(input_array[5])
    
    train = Training(new_params, dataset = dataset)
    train.run()

    print(f"\nRMSE : {train.rmse}")

    return np.array([train.rmse])

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(n_particles = 5, dimensions = 6,
                                    bounds = bounds, options = options)

stats = optimizer.optimize(target_function, iters = 300, n_processes = 5)

print(stats)

