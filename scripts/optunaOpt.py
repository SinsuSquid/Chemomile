import optuna
import torch
import numpy as np
from src.data import Dataset
from src.train import Training
from src.model import Chemomile

# Fixed parameters
BASE_PARAMS = dict(
    target = 'ESOL',
    subfrag_size = 10,
    edge_size = 3,
    out_size = 1,
    seed = 42,
    batch_size = 128,
    max_epoch = 50, # Reduced for HPO efficiency
    verbose = False,
    save = False,
)

def objective(trial):
    # Suggest hyperparameters
    hidden_size_pow = trial.suggest_int('hidden_size_pow', 4, 8) # 16 to 256
    hidden_size = 2**hidden_size_pow
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_layers = trial.suggest_int('num_layers', 2, 8)
    num_timesteps = trial.suggest_int('num_timesteps', 2, 8)
    lr_init = trial.suggest_float('lr_init', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    params = BASE_PARAMS.copy()
    params.update({
        'hidden_size': hidden_size,
        'dropout': dropout,
        'num_layers': num_layers,
        'num_timesteps': num_timesteps,
        'lr_init': lr_init,
        'gamma': gamma,
        'weight_decay': weight_decay
    })

    # Initialize Dataset (cached DATADUMP will be used)
    dataset = Dataset(
        target = params['target'],
        seed = params['seed'],
        batch_size = params['batch_size'],
        verbose = params['verbose']
    )

    # Initialize Model
    model = Chemomile(
        subfrag_size = params['subfrag_size'],
        hidden_size = params['hidden_size'],
        out_size = params['out_size'],
        edge_size = params['edge_size'],
        dropout = params['dropout'],
        num_layers = params['num_layers'],
        num_timesteps = params['num_timesteps']
    )

    # Train
    train = Training(model, params, dataset = dataset)
    
    # We'll use the validation loss for optimization
    # For Optuna, we can also implement pruning here if we wanted to
    train.run()

    return train.test_loss # Optuna will minimize this

if __name__ == "__main__":
    # Create a study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20) # 20 trials for demonstration

    print("\n" + "="*30)
    print("Hyperparameter Optimization Complete.")
    print(f"Best value: {study.best_value}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*30)
