import torch
import torch.nn.functional as F
from torch_geometric.nn.models import AttentiveFP

class Chemomile(torch.nn.Module):
    def __init__(self, 
                 subfrag_size = 4, 
                 hidden_size = 50,
                 edge_size = 1, 
                 out_size = 1, 
                 dropout = 0.20,
                 num_layers = 3,
                 num_timesteps = 3):
        super().__init__()

        self.subfrag_size = subfrag_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_size = edge_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.subfrag_level = AttentiveFP(in_channels = self.subfrag_size,
                                         hidden_channels = 2 * self.hidden_size,
                                         out_channels = self.hidden_size,
                                         edge_dim = self.edge_size,
                                         num_layers = self.num_layers,
                                         num_timesteps = self.num_timesteps,
                                         dropout = self.dropout).to(self.device)

        self.fragment_level = AttentiveFP(in_channels = self.hidden_size,
                                          hidden_channels = 2 * self.hidden_size,
                                          out_channels = self.out_size,
                                          edge_dim = self.edge_size,
                                          num_layers = self.num_layers,
                                          num_timesteps = self.num_timesteps,
                                          dropout = self.dropout).to(self.device)

        self.reset_parameters()

        return

    def reset_parameters(self):
        self.subfrag_level.reset_parameters()
        self.fragment_level.reset_parameters()

    def jtBatchMaker(self, numFrag, jt_index, jt_attr):
        batch_idx = 0
        frag_batch = []

        for n in numFrag:
            frag_batch.extend([batch_idx for _ in range(n)])
            batch_idx += 1

        frag_batch = torch.tensor(frag_batch).to(torch.long).to(self.device)

        edge_index = []; edge_attr = []; edge_idx = 0

        for frag_idx, frag_jt in enumerate(jt_index):
            for pair_idx, pair in enumerate(frag_jt):
                edge_index.append([pair[0] + edge_idx, pair[1] + edge_idx])
                edge_attr.append(jt_attr[frag_idx][pair_idx])
            edge_idx += numFrag[frag_idx]

        edge_index = torch.tensor(edge_index).t().to(torch.long).view(2,-1)
        edge_attr = torch.tensor(edge_attr).to(torch.float).view(edge_index.shape[1], -1)

        if edge_index.numel() > 0: # Sort indices
            perm = (edge_index[0] * frag_batch.shape[0] + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        return edge_index, edge_attr, frag_batch


    def forward(self, x, edge_index, edge_attr, sub_batch,
                jt_index, jt_attr, numFrag):
        jt_index, jt_attr, jt_batch = self.jtBatchMaker(numFrag, jt_index, jt_attr)

        result_subfrag = self.subfrag_level(x = x,
                                            edge_index = edge_index,
                                            edge_attr = edge_attr,
                                            batch = sub_batch)

        result_frag = self.fragment_level(x = result_subfrag,
                                          edge_index = jt_index,
                                          edge_attr = jt_attr,
                                          batch = jt_batch)

        return result_frag

if __name__ == "__main__":
    import smiles2data
    from torch_geometric.loader import DataLoader

    data = smiles2data.smiles2data('C1=CC2=C(C=C1O)C(=CN2)CCN', 1)
    loader = DataLoader([smiles2data.smiles2data('C1=CC2=C(C=C1O)C(=CN2)CCN', 1),
                         smiles2data.smiles2data('C1=CC2=C(C=C1O)C(=CN2)CCN', 1)], batch_size = 2)
    model = Chemomile()

    for data in loader:
        out = model(x = data.x,
                    edge_index = data.edge_index,
                    edge_attr = data.edge_attr,
                    sub_batch = data.sub_batch,
                    jt_index = data.jt_index,
                    jt_attr = data.jt_attr,
                    numFrag = torch.tensor(data.numFrag).view(-1, ))
