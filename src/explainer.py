import torch
from torch_geometric.loader import DataLoader

SYMBOL = dict([(1, "H"), (6, "C"), (8, "O"), (7, "N"),
               (0, "*"), (9, "F"), (15, "P"), (16, "S"),
               (17, "Cl"), (35, "Br"), (53, "I")]) # Atomic Symbol

class Explainer():
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.model.device = "cpu"
        self.model = self.model.to("cpu")

        self.original()

        return

    def original(self):
        original_loader = DataLoader([self.data])

        results = []

        with torch.no_grad():
            self.model.eval()
            for data in original_loader:
                out = self.model(x = data.x,
                                 edge_index = data.edge_index,
                                 edge_attr = data.edge_attr,
                                 sub_batch = data.sub_batch,
                                 jt_index = data.jt_index,
                                 jt_attr = data.jt_attr,
                                 numFrag = torch.tensor([data.numFrag]).view(-1, ),
                                 mol_x = data.mol_x,
                                 mol_edge_index = data.mol_edge_index,
                                 mol_edge_attr = data.mol_edge_attr,
                                 numAtom = torch.tensor([data.numAtom]).view(-1, ),
                                )
                results.append(out.item())

        self.ref = sum(results) / len(results)

        return

    def atomMask(self):
        from copy import deepcopy

        masked_list = [[pairIdx for pairIdx, pair in enumerate(self.data.mol_edge_index)
                                 if i not in pair]
                                for i in range(self.data.numAtom)]

        masked_data = []
        for i in range(self.data.numAtom):
            temp_data = deepcopy(self.data)
            temp_data.mol_edge_index = [self.data.mol_edge_index[j] for j in masked_list[i]]
            temp_data.mol_edge_attr = [self.data.mol_edge_attr[j] for j in masked_list[i]]
            masked_data.append(temp_data)

        masked_loader = DataLoader(masked_data, batch_size = 1)

        results = []

        with torch.no_grad():
            self.model.eval()
            for data in masked_loader:
                out = self.model(x = data.x,
                                 edge_index = data.edge_index,
                                 edge_attr = data.edge_attr,
                                 sub_batch = data.sub_batch,
                                 jt_index = data.jt_index,
                                 jt_attr = data.jt_attr,
                                 numFrag = data.numFrag,
                                 mol_x = data.mol_x,
                                 mol_edge_index = data.mol_edge_index,
                                 mol_edge_attr = data.mol_edge_attr,
                                 numAtom = data.numAtom,
                                )
                results.append(out.item())

        self.results = [(results[i] - self.ref) for i in range(self.data.numAtom)]
        self.results = [(self.ref - results[i]) for i in range(self.data.numAtom)]

        return self.results

    def plot(self, ax):
        anum = self.data.mol_x[:,0]
        coord = self.data.mol_x[:,-3:]
        bonds = self.data.mol_edge_index

        p = ax.scatter(coord[:,0], coord[:,1] ,coord[:,2], s = 200, c = self.results, cmap = "seismic", alpha = 0.7)

        for c in range(coord.shape[0]):
            ax.text(coord[c,0], coord[c,1], coord[c,2], SYMBOL[int(anum[c])],
                    ha = 'center', va = 'center')

        for i, j in bonds:
            ax.plot((coord[i,0], coord[j,0]),
                    (coord[i,1], coord[j,1]),
                    (coord[i,2], coord[j,2]),
                    color = 'k',
                   )
        
        ax.axis('off')
        
        return p

if __name__ == "__main__":
    import sys
    sys.path.append("/SSD2/bgkang/Chemomile")
    
    from src.smiles2data import smiles2data
    from src.model import Chemomile

    SMILES = "C1=CC2=C(C=C1O)C(=CN2)CCN"
    TRUE_VAL = 1.0

    model = Chemomile(
            subfrag_size = 12,
            hidden_size = 84,
            out_size = 1,
            edge_size = 3,
            dropout = 0.360,
            num_layers = 4,
            num_timesteps = 4,
    )

    data = smiles2data(SMILES, TRUE_VAL)
    model.eval()

    explainer = Explainer(
        model = model,
        data = data
    )
    
    print(f"Original Output : {explainer.ref}")

    score = explainer.atomMask()
    print(score)

    explainer.plot()