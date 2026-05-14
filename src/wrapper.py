import torch
import os
from src.model import Chemomile
from src.smiles2data import smiles2data
from src.explainer import Explainer

class ChemomileWrapper:
    OPTIMIZED_PARAMS = {
        'FP' : {'hidden_size' : 128, 'dropout' : 0.276, 'num_layers' : 6, 'num_timesteps' : 6},
        'AIT' : {'hidden_size' : 64, 'dropout' : 0.220, 'num_layers' : 3, 'num_timesteps' : 6},
        'HCOM' : {'hidden_size' : 128, 'dropout' : 0.230, 'num_layers' : 6, 'num_timesteps' : 5},
        'FLVL' : {'hidden_size' : 128, 'dropout' : 0.450, 'num_layers' : 5, 'num_timesteps' : 6},
        'FLVU' : {'hidden_size' : 256, 'dropout' : 0.307, 'num_layers' : 5, 'num_timesteps' : 7},
        'ESOL' : {'hidden_size' : 32, 'dropout' : 0.238, 'num_layers' : 4, 'num_timesteps' : 6},
    }

    DEFAULT_PARAMS = {
        'subfrag_size' : 10,
        'edge_size' : 3,
        'out_size' : 1,
    }

    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        self._model_cache = {}
        self.weight_status = {} # Track if weights were successfully loaded
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Wrapper initialized on {self.device}")

    def has_weights(self, target):
        model_path = os.path.join(self.model_dir, f"{target}_model.pt")
        exists = os.path.exists(model_path)
        self.weight_status[target] = exists
        return exists

    def _get_model(self, target):
        if target in self._model_cache:
            return self._model_cache[target]

        if target not in self.OPTIMIZED_PARAMS:
            raise ValueError(f"Unknown target: {target}")

        params = self.DEFAULT_PARAMS.copy()
        params.update(self.OPTIMIZED_PARAMS[target])

        model = Chemomile(
            subfrag_size=params['subfrag_size'],
            hidden_size=params['hidden_size'],
            out_size=params['out_size'],
            edge_size=params['edge_size'],
            dropout=params['dropout'],
            num_layers=params['num_layers'],
            num_timesteps=params['num_timesteps'],
        ).to(self.device)

        model_path = os.path.join(self.model_dir, f"{target}_model.pt")
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded weights for {target} from {model_path}")
                self.weight_status[target] = True
            except Exception as e:
                print(f"Error loading weights for {target}: {e}")
                self.weight_status[target] = False
        else:
            print(f"Warning: No weights found for {target} at {model_path}.")
            self.weight_status[target] = False

        model.eval()
        self._model_cache[target] = model
        return model

    def predict(self, smiles, target):
        data = smiles2data(smiles, 0)
        if data == -1:
            return None

        model = self._get_model(target)

        with torch.no_grad():
            output = model(
                x = data.x,
                edge_index = data.edge_index,
                edge_attr = data.edge_attr,
                sub_batch = data.sub_batch,
                jt_index = [data.jt_index],
                jt_attr = [data.jt_attr],
                numFrag = torch.tensor([data.numFrag]).view(-1, ),
                mol_x = data.mol_x,
                mol_edge_index = [data.mol_edge_index],
                mol_edge_attr = [data.mol_edge_attr],
                numAtom = [data.numAtom]
            )
        
        return output.item()

    def explain(self, smiles, target, method="atom_mask"):
        data = smiles2data(smiles, 0)
        if data == -1:
            return None, None

        model = self._get_model(target)
        explainer = Explainer(model, data)
        
        if method == "integrated_gradients":
            scores = explainer.integratedGradients()
        else:
            scores = explainer.atomMask()
            # Convert tensors to list of floats if necessary
            scores = [s.item() if torch.is_tensor(s) else s for s in scores]
            
        return data, scores

if __name__ == "__main__":
    # Sample Test (Requires weights to exist for full verification)
    wrapper = ChemomileWrapper()
    sample_smiles = "C1=CC2=C(C=C1O)C(=CN2)CCN"
    try:
        res = wrapper.predict(sample_smiles, "ESOL")
        print(f"Prediction for {sample_smiles} (ESOL): {res}")
    except Exception as e:
        print(f"Inference deferred/failed: {e}")
