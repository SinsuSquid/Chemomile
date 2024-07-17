if __name__ == '__main__':
    from src.data import Dataset
    from src.train import Training
    from src.model import Chemomile
    from src.ensemble import EnsembleTraining
    
    parameters = dict(
        subfrag_size = 12,
        edge_size = 3,
        out_size = 1,
        seed = 42,
        batch_size = 128,
        max_epoch = 200,
        verbose = False,
        save = False,
        
        target = 'FP',
        hidden_size = 84,
        dropout = 0.350,
        num_layers = 4,
        num_timesteps = 4,
        lr_init = 0.01,
        gamma = 0.994,
        weight_decay = 3.5E-3,
    )

    dataset = Dataset(
        target = parameters['target'],
        seed = parameters['seed'],
        batch_size = parameters['batch_size'],
        verbose = parameters['verbose']
    )

    ensemble = EnsembleTraining(
            numEnsemble = 10,
            parameters = parameters, 
            dataset = dataset)

    ensemble.run()

    print(f"metrics - target : {parameters['target']}")
    print(f"\tmae : {ensemble.mae:6.3f}")
    print(f"\trmse : {ensemble.rmse:6.3f}")
    print(f"\tmdape : {ensemble.mdape:6.3f}")
    print(f"\tr2 : {ensemble.r2:6.3f}")

    ensemble.TPPlot()
