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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.atom_encoder = torch.nn.Sequential(
                torch.nn.Linear(self.subfrag_size, self.hidden_size),
                torch.nn.BatchNorm1d(self.hidden_size),
                torch.nn.LeakyReLU()
        ).to(self.device)

        self.subfrag_level = AttentiveFP(in_channels = self.hidden_size,
                                         hidden_channels = 2 * self.hidden_size,
                                         out_channels = self.hidden_size,
                                         edge_dim = self.edge_size,
                                         num_layers = self.num_layers,
                                         num_timesteps = self.num_timesteps,
                                         dropout = self.dropout).to(self.device)

        self.fragment_level = AttentiveFP(in_channels = self.hidden_size,
                                          hidden_channels = 2 * self.hidden_size,
                                          out_channels = self.hidden_size,
                                          edge_dim = self.edge_size,
                                          num_layers = self.num_layers,
                                          num_timesteps = self.num_timesteps,
                                          dropout = self.dropout).to(self.device)

        self.molecule_level = AttentiveFP(in_channels = self.hidden_size,
                                          hidden_channels = 2 * self.hidden_size,
                                          out_channels = self.hidden_size,
                                          edge_dim = self.edge_size,
                                          num_layers = self.num_layers,
                                          num_timesteps = self.num_timesteps,
                                          dropout = self.dropout).to(self.device)

        self.fully_connected = torch.nn.Sequential(
                torch.nn.Linear(in_features = 2 * self.hidden_size,
                                out_features = self.hidden_size),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.BatchNorm1d(self.hidden_size),
                torch.nn.Linear(in_features = self.hidden_size,
                                out_features = self.out_size),
                ).to(self.device)

        self.reset_parameters()

        return

    def reset_parameters(self):
        for layer in self.atom_encoder:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()
        self.subfrag_level.reset_parameters()
        self.fragment_level.reset_parameters()
        self.molecule_level.reset_parameters()
        for layer in self.fully_connected:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()

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

    def molBatchMaker(self, numAtom, mol_edge_index, mol_edge_attr):
        batch_idx = 0
        atom_batch = []

        for n in numAtom:
            atom_batch.extend([batch_idx for _ in range(n)])
            batch_idx += 1

        atom_batch = torch.tensor(atom_batch).to(torch.long).to(self.device)

        edge_index = []; edge_attr = []; edge_idx = 0

        for mol_idx, mol_edges in enumerate(mol_edge_index):
            for pair_idx, pair in enumerate(mol_edges):
                edge_index.append([pair[0] + edge_idx, pair[1] + edge_idx])
                edge_attr.append(mol_edge_attr[mol_idx][pair_idx])
            edge_idx += numAtom[mol_idx]

        edge_index = torch.tensor(edge_index).t().to(torch.long).view(2,-1)
        edge_attr = torch.tensor(edge_attr).to(torch.float).view(edge_index.shape[1], -1)

        if edge_index.numel() > 0: # Sort indices
            perm = (edge_index[0] * atom_batch.shape[0] + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        return edge_index, edge_attr, atom_batch

    def forward(self, x, edge_index, edge_attr, sub_batch,
                jt_index, jt_attr, numFrag,
                mol_x, mol_edge_index, mol_edge_attr, numAtom):

        jt_index, jt_attr, jt_batch = self.jtBatchMaker(numFrag, jt_index, jt_attr)
        x = x.to(self.device)
        x = self.atom_encoder(x)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        sub_batch = sub_batch.to(self.device)
        result_subfrag = self.subfrag_level(x = x,
                                            edge_index = edge_index,
                                            edge_attr = edge_attr,
                                            batch = sub_batch)
        
        jt_index = jt_index.to(self.device)
        jt_attr = jt_attr.to(self.device)
        jt_batch = jt_batch.to(self.device)
        result_frag = self.fragment_level(x = result_subfrag,
                                          edge_index = jt_index,
                                          edge_attr = jt_attr,
                                          batch = jt_batch)
    
        mol_edge_index, mol_edge_attr, mol_batch = self.molBatchMaker(numAtom, mol_edge_index, mol_edge_attr)
        mol_x = mol_x.to(self.device)
        mol_x = self.atom_encoder(mol_x)
        mol_edge_index = mol_edge_index.to(self.device)
        mol_edge_attr = mol_edge_attr.to(self.device)
        mol_batch = mol_batch.to(self.device)
        result_molecule = self.molecule_level(x = mol_x,
                                              edge_index = mol_edge_index,
                                              edge_attr = mol_edge_attr,
                                              batch = mol_batch)

        result = self.fully_connected(
                torch.cat([result_molecule, result_frag], axis = 1)
                )

        return result

if __name__ == "__main__":
    import smiles2data
    from torch_geometric.loader import DataLoader

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
                    numFrag = torch.tensor(data.numFrag).view(-1, ),
                    mol_x = data.mol_x,
                    mol_edge_index = data.mol_edge_index,
                    mol_edge_attr = data.mol_edge_attr,
                    numAtom = data.numAtom)
        print(out)
