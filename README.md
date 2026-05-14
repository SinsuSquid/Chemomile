# Chemomile

![Chemomile_Logo](https://github.com/SinsuSquid/Chemomile/blob/main/icons/Chemomile_Logo.png)

"Chemomile: Explainable Multi-Level GNN Model for Combustion Property Prediction", **Beomgyu Kang** and Bong June Sung*, *J. Phys. Chem. A* 2025, 129, 1880-1889, https://doi.org/10.1021/acs.jpca.5c00380

## Modernized Architecture (2026 Upgrades)
Chemomile has been recently upgraded with several state-of-the-art features:
- **Fast Training:** Integrated **Automatic Mixed Precision (AMP)** and **Vectorized Batching** for significant GPU and CPU acceleration.
- **Advanced HPO:** Added **Optuna** for Bayesian Hyperparameter Optimization with intelligent pruning.
- **Improved Explainability:** Implemented **Integrated Gradients** for smoother and mathematically grounded atom-level attribution.
- **Optimized Data Pipeline:** Parallel SMILES-to-Graph featurization with **HDF5-compressed caching**.
- **Training Stability:** Added **Early Stopping** and **Gradient Clipping** to ensure robust convergence.

## Prerequisites
- torch >= 2.10.0
- torch\_geometric >= 2.7.0
- rdkit >= 2026.03.1
- optuna >= 4.0.0
- h5py >= 3.11.0
- joblib >= 1.4.0
- networkx >= 3.6
- rich >= 14.0.0
- scikit-learn >= 1.5.0
- tqdm >= 4.66.4
- pyswarms >= 1.3.0

## Notes
Due to copyright issue, datasets are not included in this git.
The dataset is a csv file with format:

| | SMILES | Value |
|-|------:|-------|
|0|CC|12.4|
|...|...|...|

The path for datafiles can be defined when calling `Dataset()`. Pre-processed data is cached in `data/DATADUMP/` using HDF5 for efficiency.

## How to use

Target properties can be selected in each `.py` file.

- **Single Training Cycle**
```python
python single_run.py
```

- **Bayesian Hyperparameter Optimization (Optuna)**
```python
python optunaOpt.py
```

- **Particle Swarm Optimization (PSO)**
```python
python particleSwarmOpt.py
```

- **Ensemble Training Cycle**
	- k-fold ensemble training/testing cycles (parallel)
	- returns average prediction for each fold
```python
python ensemble_run.py
```

- **Explainability (XAI)**
	- Uses **Integrated Gradients** and **Atom Masking** to visualize atomic contributions in 3D.
	- See `src/explainer.py` and the example notebooks.

## Comment
If you're not sure how to use this model, please consult [some example notebooks](./Notebooks)

## Citation

```
@article{kang2025chemomile,
    author = {Kang, Beomgyu and Sung, Bong June},
    title = {Chemomile: Explainable Multi-Level GNN Model for Combustion Property Prediction},
    journal = {The Journal of Physical Chemistry A},
    volume = {129},
    number = {7},
    pages = {1880-1889},
    year = {2025},
    doi = {10.1021/acs.jpca.5c00380},
    note ={PMID: 39927844},
    URL = {https://doi.org/10.1021/acs.jpca.5c00380},
    eprint = {https://doi.org/10.1021/acs.jpca.5c00380}
}
```
