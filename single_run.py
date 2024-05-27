if __name__ == '__main__':
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
        verbose = True,
        
        hidden_size = 30,
        dropout = 0.256,
        num_layers = 2,
        num_timesteps = 2,
        lr_init = 0.001,
        gamma = 0.994,
    )

    dataset = Dataset(
        target = parameters['target'],
        seed = parameters['seed'],
        batch_size = parameters['batch_size'],
        verbose = parameters['verbose']
    )

    train = Training(parameters, dataset = dataset)
    train.run()

    print(f"Metrics - Target : {parameters['target']}")
    print(f"\tMAE : {train.mae:6.3f}")
    print(f"\tRMSE : {train.rmse:6.3f}")
    print(f"\tMDAPE : {train.mdape:6.3f}")
    print(f"\tR2 : {train.r2:6.3f}")

    train.TPPlot()
