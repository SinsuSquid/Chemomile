if __name__ == '__main__':
    from src.data import Dataset
    from src.train import Training
    from src.model import Chemomile
    
    parameters = dict(
        subfrag_size = 12,
        edge_size = 3,
        out_size = 1,
        seed = 42,
        batch_size = 128,
        max_epoch = 200,
        verbose = True,
        save = False,
        
        target = 'RAT_INTRAPERITONEAL_LD50',
        hidden_size = 65,
        dropout = 0.266,
        num_layers = 4,
        num_timesteps = 4,
        lr_init = 0.01,
        gamma = 0.995,
        weight_decay = 1.8E-3,
    )

    model = Chemomile(
        subfrag_size = parameters['subfrag_size'],
        hidden_size = parameters['hidden_size'],
        out_size = parameters['out_size'],
        edge_size = parameters['edge_size'],
        dropout = parameters['dropout'],
        num_layers = parameters['num_layers'],
        num_timesteps = parameters['num_timesteps'],
    )

    dataset = Dataset(
        target = parameters['target'],
        seed = parameters['seed'],
        batch_size = parameters['batch_size'],
        verbose = parameters['verbose']
    )

    train = Training(model, parameters, dataset = dataset)
    train.run()

    print(f"Metrics - Target : {parameters['target']}")
    print(f"\tMAE : {train.mae:6.3f}")
    print(f"\tRMSE : {train.rmse:6.3f}")
    print(f"\tMDAPE : {train.mdape:6.3f}")
    print(f"\tR2 : {train.r2:6.3f}")

    train.TPPlot()
