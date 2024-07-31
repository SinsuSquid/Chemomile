import os
import sys
import random
import pickle
import pandas as pd
from rich.progress import track

from rdkit import RDLogger
from torch_geometric.loader import DataLoader

import src.smiles2data

ROOT = os.getcwd()

RDLogger.DisableLog('rdApp.*')

REGRESSION = ['ESOL', 'FREESOLV', 'LIPOPHILICITY']
CLASSIFICATION = []
DIPPR = ['FP', 'AIT', 'FLVL', 'FLVU', 'HCOM']
TOXICITY = ['LC50', 'MOUSE_INTRAPERITONEAL_LDLo', 'IGC50', 'RAT_INTRAPERITONEAL_LD50']
CASESTUDY = ['ORGANOSILICONE']

class Dataset():
    def __init__(self, target, seed = 42, batch_size = 1, verbose = True,
                 root = ROOT, organicOnly = False):
        self.target = target
        self.seed = seed
        self.batch_size = batch_size
        self.verbose = verbose
        self.root = root

        if self.target in REGRESSION:
            self.df = pd.read_csv(f"{self.root}/data/MOLECULENET/REGRESSION/{self.target}.csv")
        elif self.target in CLASSIFICATION:
            self.df = pd.read_csv(f"{self.root}/data/MOLECULENET/CLASSIFICATION/{self.target}.csv")
        elif self.target in DIPPR:
            self.df = pd.read_csv(f"{self.root}/data/DIPPR/{self.target}.csv")
        elif self.target in TOXICITY:
            self.df = pd.read_csv(f"{self.root}/data/TOXICITY/{self.target}.csv")
        elif self.target in CASESTUDY:
            self.df = pd.read_csv(f"{self.root}/data/CASESTUDY/{self.target}.csv")
        else:
            print("Something is wrong with the target.")
            print("Supported targets are : ")
            print(f"\tRegression : {REGRESSION}")
            print(f"\tClassification : {CLASSIFICATION}")
            print(f"\tDIPPR : {DIPPR}")
            print(f"\tTOXICITY : {TOXICITY}")
            sys.exit(-1)

        # Normalization
        self.df['Z_Value'] = (self.df['Value'] - self.df['Value'].mean()) / self.df['Value'].std()

        self.mean = self.df['Value'].mean()
        self.std = self.df['Value'].std()

        if os.path.exists(f"{root}/data/DATADUMP/{self.target}.pickle"):
            self.loadPickle()
        else:
            self.initialize()

        if organicOnly:
            self.total_set = [data for data in self.total_set if data.isOrganic]

        self.loader_initialize()

        return

    def loader_initialize(self):
        random.seed(self.seed)
        random.shuffle(self.total_set)

        self.length = len(self.total_set)

        self.training_loader = DataLoader(self.total_set[:int(0.8 * self.length)],
                                          batch_size = self.batch_size)
        self.validation_loader = DataLoader(self.total_set[int(0.8 * self.length):int(0.9 * self.length)],
                                            batch_size = self.batch_size)
        self.test_loader = DataLoader(self.total_set[int(0.9 * self.length):],
                                      batch_size = self.batch_size)
        self.total_loader = DataLoader(self.total_set, batch_size = self.batch_size)

        if self.verbose : 
            print(f"Training : {len(self.training_loader.dataset)} | Validation : {len(self.validation_loader.dataset)} | Test : {len(self.test_loader.dataset)}")
            print(f"Total : {len(self.training_loader.dataset) + len(self.validation_loader.dataset) + len(self.test_loader.dataset)}")

        return

    def loadPickle(self):
        if self.verbose : print(f"\tDataDump found for \'{self.target}\'. Loading dumped data.")

        with open(f'{self.root}/data/DATADUMP/{self.target}.pickle', 'rb') as fp:
            self.total_set = pickle.load(fp)

        return

    def initialize(self):
        if self.verbose : print(f"\tNo DataDump found for \'{self.target}\'. Creating a new one.")

        self.total_set = []

        for idx, row in track(self.df.iterrows(), total = self.df.shape[0], description = "Building DataDump ..."):
            result = src.smiles2data.smiles2data(row.SMILES, row.Z_Value)
            if (result != -1): self.total_set.append(result)

        with open(f'{self.root}/data/DATADUMP/{self.target}.pickle', 'wb') as fp:
            pickle.dump(self.total_set, fp)

        return

    def BaggingLoaders(self, k = 5):
        training_loaders = []
        validation_loaders = []

        samplePool = self.total_set[:int(0.9 * self.length)]
        sampleSize = len(samplePool)

        for _ in range(k):
            random.shuffle(samplePool)
            training_loaders.append(
                    DataLoader(samplePool[int(1.0/k * sampleSize):],
                        batch_size = self.batch_size),
                    )
            validation_loaders.append(
                    DataLoader(samplePool[:int(1.0/k * sampleSize)],
                        batch_size = self.batch_size),
                    )

        return training_loaders, validation_loaders

if __name__ == '__main__':
    import sys
    dataset = Dataset('IGC50', batch_size = 32, seed = 42)
