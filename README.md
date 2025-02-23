# Chemomile

![Chemomile_Logo](https://github.com/SinsuSquid/Chemomile/blob/main/icons/Chemomile_Logo.png)

"Chemomile: Explainable Multi-Level GNN Model for Combustion Property Prediction", **Beomgyu Kang** and Bong June Sung*, *J. Phys. Chem. A* 2025, 129, 1880-1889, https://doi.org/10.1021/acs.jpca.5c00380

## Prerequisites
- torch >= 2.3.0
- torch\_geometric >= 2.5.3
- networkx >= 3.3
- rdkit >= 2023.9.6
- rich >= 13.7.1
- scikit-learn >= 1.5.0
- tqdm >= 4.66.4
- pyswarms >= 1.3.0

## Notes
Due to copyright issue, datasets are not included in this git.
The dataset is a csv file with format:

| | SMILE | Value |
|-|------:|-------|
|0|CC|12.4|
|...|...|...|

The path for datafiles can be defined when calling Dataset()

## How to use

Target properties can be selected in each .py file.

- single training/testing cycle
```python
python single_run.py
```

- ensemble training/testing cycle
	- k-fold ensemble training/testing cycles (parallel)
	- returns averages prediction for each fold
```python
python ensemble_run.py
```

- Particle Swarm Optimization (PSO)
	- PSO is implemented with [pyswarms](https://github.com/ljvmiranda921/pyswarms)
	- each particle undergoes training/testing cycle at each iteration (parallel)
	- object function can be defined in the particleSwarmOpt.py
	- phase space boundaries are defined in `bounds`
```python
python particleSwarmOpt.py
```

## Comment
If you're not sure how to use this model, please consult [some example notebooks](./Notebooks)

## Citation

```
@article{doi:10.1021/acs.jpca.5c00380,
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