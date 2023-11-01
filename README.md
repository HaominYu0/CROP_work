# CROP
This is a PyTorch implementation of the paper: A Crystal Knowledge-enhanced Pre-training Framework for
Crystal Property Estimation.
`@author:Haomin Yu, Yanru Song, Jilin Hu, Chenjuan Guo, Bin Yang, Christian S. Jensen`

## Requirements
- arrow==0.15.4
- ase==3.22.1
- bresenham==0.2.1
- dgl==0.6.1
- dgl_cu110==0.6.1
- dill==0.3.6
- geopy==1.20.0
- ipdb==0.13.9
- matbench==0.6
- matplotlib==3.1.1
- MetPy==1.3.1
- networkx==2.7.1
- numpy==1.22.4
- p_tqdm==1.4.0
- pandas==1.4.2
- pydantic==1.10.4
- pymatgen==2022.11.7
- PyYAML==6.0
- scikit_learn==1.0.1
- scipy==1.7.3
- sympy==1.10.1
- timm==0.4.12
- torch==1.13.1
- torch_cluster==1.6.0
- torch_geometric==2.2.0
- torch_scatter==2.0.9
- torch_sparse==0.6.16
- torchvision==0.14.1
- tqdm==4.64.0

## Datasets
To pre-train the framework, we use a subset  of the Open   Materials Database (OQMD) with 817,139 material structures. 
The OQMD offers a substantial amount of unlabeled data that proves sufficient for the purpose of pre-training. This dataset includes the data from the Open   Materials Database (OQMD) and is obtained by JARVIS-Tools, an open-access software package for atomistic data-driven materials computation. It has 817,636 material structures. To ensure the quality of the dataset, we filtered out the materials with extreme formation energy that is either above 4.0 or below -5.0. We also filter out a crystal structure \footnote{https://oqmd.org/materials/entry/1339536}  that cannot be successfully loaded. The cleaned dataset contains 817,139 material structures for the pre-training task. Each structure is saved as a CIF file and contains atom type, atom coordinates, and lattice features.

## Model Training
The models have been pretrained on the CIF files from the OQMD database. In total, we aggregated 817,139 CIF files for pretraining. The pretrained models, along with their config files, are available in the repository.

To execute the pretrained model, use the following command:
```python
 python crop_pretrain.py
```
The parameters for the pretraining of model can be modified in the config.yaml

Finetuning model:

To fine-tune the pre-trained model on the JDFT2D, Dielectric, and KVRH datasets, run the following command using the Matbench_base files:

```python
 python finetune.py
```
 
To fine-tune  the pre-trained model on the Mp_shear, Mp_bulk, Jarvis_gap, Jarvis_ehull, and Mp_gap datasets, execute the following command using the Matformer_base files:
 ```python
 python finetune.py
```
