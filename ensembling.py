if __name__ == '__main__':
    import sys
    import torch
    from src.data import Dataset
    from src.train import Training
    from src.model import Chemomile

    if len(sys.argv) != 2:
        print("USAGE : python ensembling.py __model__")
        sys.exit()

    NUMENSEMBLE = 100
    PATH = sys.argv[1]
    
    parameters = dict(
        target = 'ESOL',
        subfrag_size = 12,
        edge_size = 3,
        out_size = 1,
        seed = 42,
        batch_size = 256,
        max_epoch = 200,
        verbose = False,
       
        # Please input the Optimized parameters
        hidden_size = 91,
        dropout = 0.282,
        num_layers = 2,
        num_timesteps = 4,
        lr_init = 0.01,
        gamma = 0.980,
        weight_decay = 0.01,
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
    model.load_state_dict(torch.load(PATH))
    model.eval()

print(f"# Ensembling result for {parameters['target']}")
print("#\tindex\tMAE\tRMSE\tMDAPE\tR2")

for ensemble in range(NUMENSEMBLE):
        dataset = Dataset(
            target = parameters['target'],
            seed = ensemble,
            batch_size = parameters['batch_size'],
            verbose = parameters['verbose']
        )

        train = Training(model, parameters, dataset = dataset)
        train.metrics(*train.eval())

        print(f"{ensemble:4d}\t{train.mae}\t{train.rmse}\t{train.mdape}\t{train.r2}")

