if __name__ == '__main__':
    from src.data import Dataset
    from src.train import Training
    from src.model import Chemomile
    from src.ensemble import EnsembleTraining

    optimized = {
        'FP' : {'hidden_size' : 128, 'dropout' : 0.276, 'num_layers' : 6, 'num_timesteps' : 6, 'lr_init' : 0.01, 'gamma' : 0.978, 'weight_decay' : 6.62E-3},
        'AIT' : {'hidden_size' : 64, 'dropout' : 0.220, 'num_layers' : 3, 'num_timesteps' : 6, 'lr_init' : 0.01, 'gamma' : 0.981, 'weight_decay' : 2.67E-3},
        'HCOM' : {'hidden_size' : 128, 'dropout' : 0.230, 'num_layers' : 6, 'num_timesteps' : 5, 'lr_init' : 0.001, 'gamma' : 0.979, 'weight_decay' : 3.92E-4},
        'FLVL' : {'hidden_size' : 128, 'dropout' : 0.450, 'num_layers' : 5, 'num_timesteps' : 6, 'lr_init' : 0.01, 'gamma' : 0.984, 'weight_decay' : 6.95E-4},
        'FLVU' : {'hidden_size' : 256, 'dropout' : 0.307, 'num_layers' : 5, 'num_timesteps' : 7, 'lr_init' : 0.01, 'gamma' : 0.983, 'weight_decay' : 1.26E-2},
        'ESOL' : {'hidden_size' : 32, 'dropout' : 0.238, 'num_layers' : 4, 'num_timesteps' : 6, 'lr_init' : 0.001, 'gamma' : 0.993, 'weight_decay' : 1.23E-3},
    }
    
    parameters = dict(
        subfrag_size = 10,
        edge_size = 3,
        out_size = 1,
        seed = 1217,
        batch_size = 128,
        max_epoch = 200,
        verbose = False,
        save = False,
        
        target = 'FLVU',
    )

    parameters = parameters | optimized[parameters['target']]

    dataset = Dataset(
        target = parameters['target'],
        seed = parameters['seed'],
        batch_size = parameters['batch_size'],
        verbose = parameters['verbose']
    )

    ensemble = EnsembleTraining(
            numEnsemble = 5,
            parameters = parameters, 
            dataset = dataset)

    ensemble.run()

    mae, rmse, mape, r2 = ensemble.ensemble_metrics()

    print(f"metrics - target : {parameters['target']}")
    print(f"\tmae : {mae.mean():6.3f} ± {mae.std():6.3f}")
    print(f"\trmse : {rmse.mean():6.3f} ± {rmse.std():6.3f}")
    print(f"\tmape : {mape.mean():6.3f} ± {mape.std():6.3f}")
    print(f"\tr2 : {r2.mean():6.3f} ± {r2.std():6.3f}")

    # ensemble.TPPlot()
