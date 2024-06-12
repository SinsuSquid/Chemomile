import numpy as np
import torch
import rdkit

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

BONDTYPE = [
    rdkit.Chem.rdchem.BondType.UNSPECIFIED,
    rdkit.Chem.rdchem.BondType.SINGLE,
    rdkit.Chem.rdchem.BondType.DOUBLE,
    rdkit.Chem.rdchem.BondType.TRIPLE,
    rdkit.Chem.rdchem.BondType.QUADRUPLE,
    rdkit.Chem.rdchem.BondType.QUINTUPLE,
    rdkit.Chem.rdchem.BondType.HEXTUPLE,
    rdkit.Chem.rdchem.BondType.ONEANDAHALF,
    rdkit.Chem.rdchem.BondType.TWOANDAHALF,
    rdkit.Chem.rdchem.BondType.THREEANDAHALF,
    rdkit.Chem.rdchem.BondType.FOURANDAHALF,
    rdkit.Chem.rdchem.BondType.FIVEANDAHALF,
    rdkit.Chem.rdchem.BondType.AROMATIC,
    rdkit.Chem.rdchem.BondType.IONIC,
    rdkit.Chem.rdchem.BondType.HYDROGEN,
    rdkit.Chem.rdchem.BondType.THREECENTER,
    rdkit.Chem.rdchem.BondType.DATIVEONE,
    rdkit.Chem.rdchem.BondType.DATIVE,
    rdkit.Chem.rdchem.BondType.DATIVEL,
    rdkit.Chem.rdchem.BondType.DATIVER,
    rdkit.Chem.rdchem.BondType.OTHER,
    rdkit.Chem.rdchem.BondType.ZERO,
]

STEREO = [
    rdkit.Chem.rdchem.BondStereo.STEREONONE,
    rdkit.Chem.rdchem.BondStereo.STEREOANY,
    rdkit.Chem.rdchem.BondStereo.STEREOZ,
    rdkit.Chem.rdchem.BondStereo.STEREOE,
    rdkit.Chem.rdchem.BondStereo.STEREOCIS,
    rdkit.Chem.rdchem.BondStereo.STEREOTRANS,
]

CHIRALITY = [
    rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdkit.Chem.rdchem.ChiralType.CHI_OTHER,
    rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
    rdkit.Chem.rdchem.ChiralType.CHI_ALLENE,
    rdkit.Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
    rdkit.Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    rdkit.Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,
]

HYBRIDIZATION = [
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.OTHER,
]

ORGANICATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

def smiles2data(smiles, y, seed = 42):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize = True) # default mol object
        acyclic_single = [idx for idx, bond in enumerate(mol.GetBonds())
                               if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE) and (not bond.IsInRing())]

        mol_H = Chem.AddHs(mol) # molecule with Hydrogens
        AllChem.EmbedMultipleConfs(mol, 10)
        optResult = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)

        fragments_mol = []
        fragments_atom = []

        if len(acyclic_single) == 0: # molecule with no fragments
            fragments_mol.append(mol_H)
        else: # molecule with more than 2 fragments
            mol_div = Chem.FragmentOnBonds(mol, acyclic_single, addDummies = True)
            fragments_mol = Chem.GetMolFrags(mol_div, asMols = True) # mol objects
            fragments_atom = Chem.GetMolFrags(mol_div, asMols = False) # tuple of atom index

        x, edge_index, edge_attr, sub_batch = subfragment_data(fragments_mol, seed = seed)
        jt_index, jt_attr = junction_tree(mol, fragments_atom, acyclic_single)
        mol_x, mol_edge_index, mol_edge_attr, mol_batch = subfragment_data([mol_H], seed = seed)
        numFrag = len(fragments_mol)
        numAtom = mol_x.shape[0]
        
        # NOTE : Unwanted re-indexing happens while Using DataLoader.
        #        So, we'll handle this problem within the model, with function named "batchMaker"

        isOrganic = checkOrganic(mol)

        data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, sub_batch = sub_batch, numFrag = numFrag, 
                    jt_index = jt_index, jt_attr = jt_attr,
                    mol_x = mol_x, mol_edge_index = mol_edge_index.T.tolist(), mol_edge_attr = mol_edge_attr.tolist(), numAtom = numAtom,
                    y = y, smiles = smiles, isOrganic = isOrganic)

        # printData(data)

    except (RuntimeError, ValueError, AttributeError) as e:
        print(smiles)
        print(e)

        data = -1

    return data

def checkOrganic(mol):
    result = True
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ORGANICATOMS:
            result = False
    return result

def printData(data):
    for key in data.keys():
        try:
            print(f"{key}.shape", data[key].shape)
            print(data[key])
        except:
            print(f"{key}", data[key])
    return 

def junction_tree(mol, fragments_atom, acyclic_single):
    jt_index = []; jt_attr = []

    for bond_index in acyclic_single:
        bond = mol.GetBondWithIdx(bond_index)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        frag_pair = [100, 100]
        
        edge_data = [BONDTYPE.index(bond.GetBondType())]

        # add edge_data here
        edge_data += [STEREO.index(bond.GetStereo())]
        edge_data += [1 if bond.GetIsConjugated() else 0]

        for frag_idx, frag in enumerate(fragments_atom):
            if i in frag: frag_pair[0] = frag_idx
            if j in frag: frag_pair[1] = frag_idx

        jt_index.append(frag_pair)
        jt_index.append(frag_pair[::-1])
        jt_attr += [edge_data]
        jt_attr += [edge_data]

    
    return jt_index, jt_attr

def subfragment_data(fragments_mol, seed = 42):
    x = []; edge_indices = []; edge_attr = []; batch = []
    batch_idx = 0; num_atoms = 0

    for frag in fragments_mol:
        frag = Chem.AddHs(frag)
        Chem.SanitizeMol(frag)
        AllChem.EmbedMultipleConfs(frag, 10)
        optResult = AllChem.MMFFOptimizeMoleculeConfs(frag, maxIters=2000)
        minIdx = 100000; minEng = 10000.0
        for idx, (converged, energy) in enumerate(optResult):
            if (energy < minEng): 
                minIdx = idx
                minEng = energy
        conf = frag.GetConformer(minIdx)

        position = np.array(conf.GetPositions())
        com = position.mean(axis = 0)

        position = position - com # relative position from the centre of mass

        for idx, atom in enumerate(frag.GetAtoms()):
            atom_data = [atom.GetAtomicNum()]

            # add atom_data here

            atom_data += [CHIRALITY.index(atom.GetChiralTag())]
            atom_data += [atom.GetTotalDegree()]
            atom_data += [atom.GetFormalCharge()]
            atom_data += [atom.GetTotalNumHs()]
            atom_data += [atom.GetNumRadicalElectrons()]
            atom_data += [HYBRIDIZATION.index(atom.GetHybridization())]
            atom_data += [1 if atom.GetIsAromatic() else 0]
            atom_data += [1 if atom.IsInRing() else 0]
            
            atom_data += tuple(position[idx])

            x.append(atom_data)
            batch.append(batch_idx)

        batch_idx += 1

        for bond in frag.GetBonds():
            i = bond.GetBeginAtomIdx() + num_atoms
            j = bond.GetEndAtomIdx() + num_atoms

            edge_data = [BONDTYPE.index(bond.GetBondType())]

            # add edge_data here

            edge_data += [STEREO.index(bond.GetStereo())]
            edge_data += [1 if bond.GetIsConjugated() else 0]

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

    data = smiles2data(SMILES,y)
    print(data.x)
