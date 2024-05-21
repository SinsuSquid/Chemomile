if __name__ == '__main__':
    from src.data import Dataset
    from src.train import Training
    
    parameters = dict(
        target = 'FLVL',
        subfrag_size = 4,
        edge_size = 1,
        out_size = 1,
        seed = 42,
        batch_size = 32,
        max_epoch = 200,
        verbose = False,
        
        hidden_size = 50,
        dropout = 0.20,
        num_layers = 3,
        num_timesteps = 3,
        lr_init = 0.001,
        gamma = 0.998,
    )

    dataset = Dataset(
        target = parameters['target'],
        seed = parameters['seed'],
        batch_size = parameters['batch_size'],
        verbose = parameters['verbose']
    )

    train = Training(parameters, dataset = dataset)
    train.run()

    print("Metrics")
    print(f"MAE : {train.mae:6.3f}")
    print(f"RMSE : {train.rmse:6.3f}")
    print(f"MDAPE : {train.mdape:6.3f}")
    print(f"R2 : {train.r2:6.3f}")