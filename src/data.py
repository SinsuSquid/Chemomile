import os
import random
import pickle
import pandas as pd
import tqdm

from rdkit import RDLogger
from torch_geometric.loader import DataLoader

import src.smiles2data

ROOT = os.getcwd()

RDLogger.DisableLog('rdApp.*')

MOLECULENET = ['ESOL', 'FREESOLV', 'LIPOPHILICITY']
DIPPR = ['FP', 'AIT', 'FLVL', 'FLVU', 'HCOM']

class Dataset():
    def __init__(self, target, seed, batch_size, verbose = True):
        self.target = target
        self.seed = seed
        self.batch_size = batch_size
        self.verbose = verbose

        if self.target in MOLECULENET:
            self.df = pd.read_csv(f"{ROOT}/data/MOLECULENET/{self.target}.csv")
        elif self.target in DIPPR:
            self.df = pd.read_csv(f"{ROOT}/data/DIPPR/{self.target}.csv")
        else:
            print("Something is wrong with the target.")
            print("Supported targets are : ")
            print(f"\tMoleculeNet : {MOLECULENET}")
            print(f"\tDIPPR : {DIPPR}")
            exit(-1)

        # Normalization
        self.df['Z_Value'] = (self.df['Value'] - self.df['Value'].mean()) / self.df['Value'].std()

        if os.path.exists(f"{ROOT}/data/DATADUMP/{self.target}.pickle"):
            self.loadPickle()
        else:
            self.initialize()

        self.loader_initialize()

        return

    def loader_initialize(self):
        random.seed(self.seed)
        random.shuffle(self.total_set)

        length = len(self.total_set)

        self.training_loader = DataLoader(self.total_set[:int(0.8 * length)],
                                          batch_size = self.batch_size)
        self.validation_loader = DataLoader(self.total_set[int(0.8 * length):int(0.9 * length)],
                                            batch_size = self.batch_size)
        self.test_loader = DataLoader(self.total_set[int(0.9 * length):],
                                      batch_size = self.batch_size)

        if self.verbose : 
            print(f"Training : {len(self.training_loader.dataset)} | Validation : {len(self.validation_loader.dataset)} | Test : {len(self.test_loader.dataset)}")
            print(f"Total : {len(self.training_loader.dataset) + len(self.validation_loader.dataset) + len(self.test_loader.dataset)}")

        return

    def loadPickle(self):
        if self.verbose : print(f"\tDataDump found for \'{self.target}\'. Loading dumped data.")

        with open(f'{ROOT}/data/DATADUMP/{self.target}.pickle', 'rb') as fp:
            self.total_set = pickle.load(fp)

        return

    def initialize(self):
        if self.verbose : print(f"\tNo DataDump found for \'{self.target}\'. Creating a new one.")

        self.total_set = []

        for idx, row in tqdm.tqdm(self.df.iterrows(), total = self.df.shape[0]):
            result = src.smiles2data.smiles2data(row.SMILES, row.Z_Value)
            if (result != -1): self.total_set.append(result)

        with open(f'{ROOT}/data/DATADUMP/{self.target}.pickle', 'wb') as fp:
            pickle.dump(self.total_set, fp)

        return

if __name__ == '__main__':
    dataset = Dataset('ESOL', batch_size = 32, seed = 42)
