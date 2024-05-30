import sys
import numpy as np
import torch

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

from src.data import Dataset
from src.smiles2data import smiles2data
from src.model import Chemomile
from torch_geometric.loader import DataLoader

if len(sys.argv) != 3:
    print("USAGE : python explain.py \"__model_path__\" \"__SMILES__\"")
    exit()
else:
    PATH = sys.argv[1]
    SMILES = sys.argv[2]

parameters = dict(
        target = "FP",
        subfrag_size = 4,
        hidden_size = 72,
        out_size = 1,
        edge_size = 1,
        dropout = 0.208,
        num_layers = 3,
        num_timesteps = 2,
        )

if (parameters['target'] not in PATH):
    print("Model name does not contain %s, is it a right model ?" % parameters['target'])
    exit()


dataset = Dataset(target = parameters['target'], seed = 42, batch_size = 32)

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

loader = DataLoader([smiles2data(SMILES, 64.0 + 273.15)])
data = [data for data in loader][0]

out = model(x = data.x,
            edge_index = data.edge_index,
            edge_attr = data.edge_attr,
            sub_batch = data.sub_batch,
            jt_index = data.jt_index,
            jt_attr = data.jt_attr,
            numFrag = torch.tensor([data.numFrag]).view(-1,))
result = out.item() * dataset.df['Value'].std() + dataset.df['Value'].mean()

print(f"SMILES : {SMILES} | True : {data.y.item():.6f} | Predicted : {result:.6f}")

explainer = Explainer(
        model = model,
        algorithm = GNNExplainer(epochs = 500),
        explanation_type = 'model',
        model_config = dict(
            mode = 'regression',
            task_level = 'graph',
            return_type = 'raw',
            ),
        node_mask_type = 'object',
        )

explanation = explainer(
        x = data.x,
        edge_index = data.edge_index,
        edge_attr = data.edge_attr,
        sub_batch = data.sub_batch,
        jt_index = data.jt_index,
        jt_attr = data.jt_attr,
        numFrag = torch.tensor([data.numFrag]).view(-1, ))

node_importance = [explanation.node_mask[i][0].item() for i in range(explanation.node_mask.shape[0])]

for idx, atom in enumerate(data.x):
    print(f"AtomicNum : {atom[0]} Importance : {node_importance[idx]}")

print(data.x)
print(data.edge_index)
print(data.sub_batch)
print(node_importance)
