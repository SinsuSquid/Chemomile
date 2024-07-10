import os
import numpy as np
import torch
from src.model import Chemomile
from joblib import Parallel, delayed

class EnsembleTraining():
    def __init__(self, numEnsemble, parameters, dataset, root = "./Model"):
        import datetime

        self.numEnsemble = numEnsemble
        self.parameters = parameters
        self.dataset = dataset
        self.root = root

        self.training_loaders, self.validation_loaders = \
                self.dataset.BaggingLoaders(k = self.numEnsemble)
        self.test_loader = self.dataset.test_loader
        self.total_loader = self.dataset.total_loader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.models = [
                    Chemomile(
                        subfrag_size = self.parameters['subfrag_size'],
                        hidden_size = self.parameters['hidden_size'],
                        out_size = self.parameters['out_size'],
                        edge_size = self.parameters['edge_size'],
                        dropout = self.parameters['dropout'],
                        num_layers = self.parameters['num_layers'],
                        num_timesteps = self.parameters['num_timesteps'],
                        ) for i in range(self.numEnsemble)
                ]

        self.loss = torch.nn.MSELoss()
        self.optimizers = [
            torch.optim.Adam(
                m.parameters(),
                lr = self.parameters['lr_init'],
                weight_decay = self.parameters['weight_decay'],
            ) for m in self.models
        ]
        self.schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                o,
                gamma = self.parameters['gamma']
            ) for o in self.optimizers
        ]

        self.ensembles = [
            item for item in zip(self.models, self.optimizers, self.schedulers, self.training_loaders, self.validation_loaders)
        ]

        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        return

    def train(self, model, optim, training_loader):
        model.train()

        training_loss = 0

        for data in training_loader:
            out = model(x = data.x,
                        edge_index = data.edge_index,
                        edge_attr = data.edge_attr,
                        sub_batch = data.sub_batch,
                        jt_index = data.jt_index,
                        jt_attr = data.jt_attr,
                        numFrag = data.numFrag,
                        mol_x = data.mol_x,
                        mol_edge_index = data.mol_edge_index,
                        mol_edge_attr = data.mol_edge_attr,
                        numAtom = data.numAtom)

            loss_obj = self.loss(out.flatten(), data.y.to(self.device))
            training_loss += loss_obj.mean()
            optim.zero_grad()
            loss_obj.backward()
            optim.step()

        training_loss = training_loss / len(training_loader)
        
        return training_loss

    def validation(self, model, validation_loader):
        model.eval()

        validation_loss = 0

        for data in validation_loader:
            out = model(x = data.x,
                        edge_index = data.edge_index,
                        edge_attr = data.edge_attr,
                        sub_batch = data.sub_batch,
                        jt_index = data.jt_index,
                        jt_attr = data.jt_attr,
                        numFrag = data.numFrag,
                        mol_x = data.mol_x,
                        mol_edge_index = data.mol_edge_index,
                        mol_edge_attr = data.mol_edge_attr,
                        numAtom = data.numAtom)

            loss_obj = self.loss(out.flatten(), data.y.to(self.device))
            validation_loss += loss_obj.mean()

        validation_loss = validation_loss / len(validation_loader)
            
        return validation_loss

    def perEnsemble(self, model, optim, scheduler, training_loader, validation_loader):
        best_model = model
        
        iterator = range(self.parameters['max_epoch'])

        valLoss_min = 1000000000000000

        for epoch in iterator:
            training_loss = self.train(model, optim, training_loader)
            validation_loss = self.validation(model, validation_loader)

            scheduler.step()

            if validation_loss < valLoss_min:
                best_model = model
                valLoss_min = validation_loss

        return best_model

    def run(self):
        self.best_models = []
        with Parallel(n_jobs = self.numEnsemble) as parallel:
            rets = parallel(
                    delayed(self.perEnsemble)(
                        model,
                        optim,
                        scheduler,
                        training_loader,
                        validation_loader
                    ) 
                    for idx, (model, optim, scheduler, training_loader, validation_loader) in enumerate(self.ensembles)
            )
            for best_model in rets:
                self.best_models.append(best_model)

        self.eval()
        
        return 

    def eval(self):
        self.trues = []; self.preds = []
        for idx, bm in enumerate(self.best_models):
            bm.eval()
            self.preds.append([])
            self.trues.append([])

            for data in self.test_loader:
                out = bm(x = data.x,
                            edge_index = data.edge_index,
                            edge_attr = data.edge_attr,
                            sub_batch = data.sub_batch,
                            jt_index = data.jt_index,
                            jt_attr = data.jt_attr,
                            numFrag = data.numFrag,
                            mol_x = data.mol_x,
                            mol_edge_index = data.mol_edge_index,
                            mol_edge_attr = data.mol_edge_attr,
                            numAtom = data.numAtom)

                self.preds[idx].append(out.to('cpu').detach().flatten().numpy())
                self.trues[idx].append(data.y.numpy())
                
            self.preds[idx] = np.concatenate(self.preds[idx]) * self.dataset.std + self.dataset.mean
            self.trues[idx] = np.concatenate(self.trues[idx]) * self.dataset.std + self.dataset.mean

        self.true = np.stack(self.trues).mean(axis = 0)
        self.pred = np.stack(self.preds).mean(axis = 0)

        self.metrics()

        return self.true, self.pred
        
    def metrics(self):
        from sklearn.metrics import r2_score
        
        self.mae = np.abs(self.true - self.pred).mean()
        self.rmse = np.sqrt(np.power(self.true - self.pred, 2).mean())
        self.mdape = np.abs((self.true - self.pred) / self.true  * 100).mean()
        self.r2 = r2_score(self.true, self.pred)

        return self.mae, self.rmse, self.mdape, self.r2

    def TPPlot(self, figsize = (8,6), dpi = 200, color = '#92A8D1'):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = plt.gca()
        
        ax.set_title(self.parameters['target'])
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

        ax.plot([self.true.min(), self.true.max()], [self.true.min(), self.true.max()], color = 'k', ls = '--')
        ax.scatter(self.true, self.pred, s = 100, color = color, label = "Averaged")

        for idx, (true, pred) in enumerate(zip(self.trues, self.preds)):
            ax.scatter(true, pred, s = 5, alpha = 0.2, label = f"ensemble{idx}")

        annot = f"MAE   : {self.mae:>8.3f}\nRMSE  : {self.rmse:>8.3f}\nMDAPE : {self.mdape:>8.3f}\nR$^2$ : {self.r2:>6.3f}"
        ax.annotate(annot, xy = (0.6, 0.1), xycoords = 'axes fraction')
        ax.legend()

        plt.show()

        return
