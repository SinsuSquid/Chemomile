import torch
import datetime
from src.model import Chemomile

class Training():
    def __init__(self, parameters, dataset):
        torch.manual_seed(seed = parameters['seed'])

        self.parameters = parameters
        self.dataset = dataset
        
        self.training_loader = self.dataset.training_loader
        self.validation_loader = self.dataset.validation_loader
        self.test_loader = self.dataset.test_loader

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = Chemomile(
            subfrag_size = self.parameters['subfrag_size'],
            hidden_size = self.parameters['hidden_size'],
            out_size = self.parameters['out_size'],
            edge_size = self.parameters['edge_size'],
            dropout = self.parameters['dropout'],
            num_layers = self.parameters['num_layers'],
            num_timesteps = self.parameters['num_timesteps']
        )
        
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.loss = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.parameters['lr_init'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma = self.parameters['gamma'])

        self.history = {'training' : [], 'validation' : [], 'lr' : []}

        return

    def eval(self):
        import numpy as np
        
        best_model = Chemomile(
            subfrag_size = self.parameters['subfrag_size'],
            hidden_size = self.parameters['hidden_size'],
            out_size = self.parameters['out_size'],
            edge_size = self.parameters['edge_size'],
            dropout = self.parameters['dropout'],
            num_layers = self.parameters['num_layers'],
            num_timesteps = self.parameters['num_timesteps']
        )
        best_model.load_state_dict(torch.load(f"./Model/{self.parameters['target']}-{self.timestamp}"))
        best_model.eval()
        
        test_loss = 0; true = []; pred = []
        
        for data in self.test_loader:
            out = best_model(x = data.x, 
                             edge_index = data.edge_index, 
                             edge_attr = data.edge_attr,
                             sub_batch = data.sub_batch, 
                             jt_index = data.jt_index,
                             jt_attr = data.jt_attr,
                             numFrag = data.numFrag)
            loss_obj = self.loss(out.flatten(), data.y.to(self.device))
        
            test_loss += loss_obj.mean()
        
            true.append(data.y.numpy())
            pred.append(out.to('cpu').detach().flatten().numpy())
        
        test_loss = test_loss / len(self.test_loader)
        
        true = np.concatenate(true) * self.dataset.df['Value'].std() + self.dataset.df['Value'].mean()
        pred = np.concatenate(pred) * self.dataset.df['Value'].std() + self.dataset.df['Value'].mean()

        return true, pred

    def train(self):
        self.model.train()
        
        training_loss = 0
        for data in self.training_loader:
            out = self.model(x = data.x, 
                             edge_index = data.edge_index, 
                             edge_attr = data.edge_attr,
                             sub_batch = data.sub_batch, 
                             jt_index = data.jt_index,
                             jt_attr = data.jt_attr,
                             numFrag = data.numFrag)
    
            loss_obj = self.loss(out.flatten(), data.y.to(self.device))
            training_loss += loss_obj.mean()
            self.optim.zero_grad()
            loss_obj.backward()
            self.optim.step()
    
        training_loss = training_loss / len(self.training_loader)
        self.history['training'].append(training_loss.item())

        return training_loss

    def validation(self):
        self.model.eval()

        validation_loss = 0
        for data in self.validation_loader:
            out = self.model(x = data.x, 
                             edge_index = data.edge_index, 
                             edge_attr = data.edge_attr,
                             sub_batch = data.sub_batch, 
                             jt_index = data.jt_index,
                             jt_attr = data.jt_attr,
                             numFrag = data.numFrag)
    
            loss_obj = self.loss(out.flatten(), data.y.to(self.device))
            validation_loss += loss_obj.mean()
    
        validation_loss = validation_loss / len(self.validation_loader)
        self.history['validation'].append(validation_loss.item())

        return validation_loss

    def run(self):
        from rich.progress import track

        if self.parameters['verbose'] : iterator = track(range(self.parameters['max_epoch']), description = "Training in process ...")
        else : iterator = range(self.parameters['max_epoch'])
        
        valLoss_min = 100000000000000
        for epoch in iterator:
            self.train()
            self.validation()

            self.history['lr'].append(self.scheduler.get_last_lr()[0])
        
            self.scheduler.step()

            if (self.parameters['verbose']):
                print(f"| Epoch : {epoch:>4d} | Trn. Loss : {self.history['training'][-1]:.3e} | Val. Loss : {self.history['validation'][-1]:.3e} | LR : {self.history['lr'][-1]:.3e} |")
            
            if (self.history['validation'][-1] < valLoss_min):
                if self.parameters['verbose'] : print(f"\tSaving the best model with valLoss : {self.history['validation'][-1]:.3f}")
                torch.save(self.model.state_dict(), f"./Model/{self.parameters['target']}-{self.timestamp}")
                valLoss_min = self.history['validation'][-1]

        true, pred = self.eval()
        self.metrics(true, pred)

        return

    def lossCurve(self, figsize = (8,6), dpi = 200):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax1 = plt.gca()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('loss')
        ax1.set_yscale('log')

        ax1.plot(list(range(self.parameters['max_epoch'])), self.history['training'],
                 label = 'Training', color = 'C0', lw = 5, alpha = 0.75)
        ax1.plot(list(range(self.parameters['max_epoch'])), self.history['validation'],
                 label = 'Validation', color = 'C1', lw = 5, alpha = 0.75)

        ax1.legend()

        ax2 = ax1.twinx()
        ax2.set_ylim((0, self.parameters['lr_init'] * 1.1))
        ax2.set_ylabel("Learning Rate", color = 'C2')
        ax2.tick_params(axis = 'y', color = 'C2', labelcolor = 'C2')
        ax2.plot(list(range(self.parameters['max_epoch'])), self.history['lr'],
                 label = 'Learning Rate', color = 'C2', lw = 5, alpha = 0.75)

        plt.show()

        return

    def metrics(self, true, pred):
        import numpy as np
        from sklearn.metrics import r2_score

        self.mae = np.abs(true - pred).mean()
        self.rmse = np.sqrt(np.power(true - pred, 2).mean())
        self.mdape = np.abs((true - pred) / true  * 100).mean()
        self.r2 = r2_score(true, pred)

        return self.mae, self.rmse, self.mdape, self.r2

    def TPPlot(self, figsize = (6,6), dpi = 200, color = '#92A8D1'):
        import matplotlib.pyplot as plt

        true, pred = self.eval()
        
        figure = plt.figure(figsize = figsize, dpi = dpi)
        ax = plt.gca()

        ax.set_title(self.parameters['target'])
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

        ax.plot([true.min(), true.max()], [true.min(), true.max()], color = 'k', ls = '--')
        ax.scatter(true, pred, alpha = 0.75, color = color)

        annot = f"MAE   : {self.mae:>8.3f}\nRMSE  : {self.rmse:>8.3f}\nMDAPE : {self.mdape:>8.3f}\nR$^2$ : {self.r2:>6.3f}"
        ax.annotate(annot, xy = (0.6, 0.1), xycoords = 'axes fraction')

        plt.show()

        return

