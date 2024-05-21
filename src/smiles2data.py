import numpy as np
import torch
import rdkit

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

BONDTYPES = [
    rdkit.Chem.rdchem.BondType.SINGLE,
    rdkit.Chem.rdchem.BondType.DOUBLE,
    rdkit.Chem.rdchem.BondType.TRIPLE,
    rdkit.Chem.rdchem.BondType.AROMATIC,
]

def smiles2data(smiles, y, seed = 42):
    try:
        mol_def = Chem.MolFromSmiles(smiles, sanitize = True) # default mol object
        acyclic_single = [idx for idx, bond in enumerate(mol_def.GetBonds())
                               if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE) and (not bond.IsInRing())]

        mol = Chem.AddHs(mol_def)
        # AllChem.EmbedMolecule(mol, AllChem.ETKDG(), randomSeed = seed)
        AllChem.EmbedMolecule(mol, randomSeed = seed)

        fragments_mol = []
        fragments_atom = []

        if len(acyclic_single) == 0: # molecule with no fragments
            fragments_mol.append(mol)
        else: # molecule with more than 2 fragments
            mol = Chem.FragmentOnBonds(mol, acyclic_single, addDummies = True)
            fragments_mol = Chem.GetMolFrags(mol, asMols = True) # mol objects
            fragments_atom = Chem.GetMolFrags(mol, asMols = False) # tuple of atom index

        x, edge_index, edge_attr, sub_batch = subfragment_data(fragments_mol, seed = seed)
        numFrag = len(fragments_mol)
        jt_index, jt_attr = junction_tree(mol_def, fragments_atom, acyclic_single)

        data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr,
                    sub_batch = sub_batch, numFrag = numFrag, 
                    jt_index = jt_index, jt_attr = jt_attr,
                    y = y, smiles = smiles)

        # printData(data)

    except (RuntimeError, ValueError, AttributeError) as e:
        print(smiles)
        print(e)

        data = -1

    return data

def printData(data):
    for key in data.keys():
        try:
            print(f"{key}.shape", data[key].shape)
            print(data[key])
        except:
            print(f"{key}", data[key])
    return 

def junction_tree(mol_def, fragments_atom, acyclic_single):
    jt_index = []; jt_attr = []

    for bond_index in acyclic_single:
        bond = mol_def.GetBondWithIdx(bond_index)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        frag_pair = [100, 100]
        
        edge_data = [BONDTYPES.index(bond.GetBondType())]

        # add edge_data here

        for frag_idx, frag in enumerate(fragments_atom):
            if i in frag: frag_pair[0] = frag_idx
            if j in frag: frag_pair[1] = frag_idx

        jt_index.append(frag_pair)
        jt_index.append(frag_pair[::-1])
        jt_attr += [edge_data]
        jt_attr += [edge_data]

    # NOTE : If we make jt_index & jt_attr as torch.tensor, unwanted re-indexing during DataLoader.
    #        So, we'll handle this problem within the model, with function named "batchMaker"
    
    return jt_index, jt_attr

def subfragment_data(fragments_mol, seed = 42):
    x = []; edge_indices = []; edge_attr = []; batch = []
    batch_idx = 0; num_atoms = 0

    for frag in fragments_mol:
        frag = Chem.AddHs(frag)
        # AllChem.EmbedMolecule(frag, AllChem.ETKDG(), randomSeed = seed)
        AllChem.EmbedMolecule(frag, randomSeed = seed)

        position = np.array(frag.GetConformer().GetPositions())
        com = position.mean(axis = 0)

        position = position - com # relative position from the centre of mass

        for idx, atom in enumerate(frag.GetAtoms()):
            atom_data = [atom.GetAtomicNum()]

            # add atom_data here
            
            atom_data += tuple(position[idx])
            x.append(atom_data)
            batch.append(batch_idx)

        batch_idx += 1

        for bond in frag.GetBonds():
            i = bond.GetBeginAtomIdx() + num_atoms
            j = bond.GetEndAtomIdx() + num_atoms

            edge_data = [BONDTYPES.index(bond.GetBondType())]

            # add edge_data here

            edge_indices += [[i, j], [j, i]]
            edge_attr += [edge_data, edge_data]

        num_atoms += frag.GetNumAtoms()

    # torch-fy & sort data
    x = torch.tensor(x).to(torch.float)
    edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2,-1)
    edge_attr = torch.tensor(edge_attr).to(torch.long)
    batch = torch.tensor(batch).to(torch.long)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return x, edge_index, edge_attr, batch
        

if __name__ == '__main__':
    SMILES = "C1=CC2=C(C=C1O)C(=CN2)CCN"
    y = 1.00

    smiles2data(SMILES,y)
