# CROP
This is a PyTorch implementation of the paper:A Crystal Knowledge-enhanced Pre-training Framework for
Crystal Property Estimation.
`@author:Haomin Yu, Yanru Song, Jilin Hu, Chenjuan Guo, Bin Yang, Christian S. Jensen`

## Requirements
arrow==0.15.4
ase==3.22.1
bresenham==0.2.1
dgl==0.6.1
dgl_cu110==0.6.1
dill==0.3.6
geopy==1.20.0
ipdb==0.13.9
matbench==0.6
matplotlib==3.1.1
MetPy==1.3.1
networkx==2.7.1
numpy==1.22.4
p_tqdm==1.4.0
pandas==1.4.2
pydantic==1.10.4
pymatgen==2022.11.7
PyYAML==6.0
PyYAML==6.0.1
scikit_learn==1.0.1
scipy==1.7.3
sympy==1.10.1
timm==0.4.12
torch==1.13.1
torch_cluster==1.6.0
torch_geometric==2.2.0
torch_scatter==2.0.9
torch_scatter==2.1.0
torch_sparse==0.6.12
torch_sparse==0.6.16
torchvision==0.14.1
tqdm==4.64.0

## Model Training
The models have been pretrained on the cif files in the OQMD database. In total we aggregate 817,139  cif files for the pretraining. The pretrained models along along with their config files are made available in the repository. 

To run the pretrained model run the command
```python
 python crop_pretrain.py
```
The parameters for the pretraining of model can be modified in the config.yaml

Finetuning model:
For Finetuning the model, we initialize with the pre-trained weights and finetune it for the downstream task. 
```python
 python finetune.py
```

To train it on the JDFT2D, Dielectric, and KVRH datasets, the run the finetuning on files Matbench_base 

```python
 python finetune.py
```
 
To train it on the Mp_shear, Mp_bulk, Jarvis_gap, Jarvis_ehull and Mp_gap datasets, the run the finetuning on files Matformer_base

 ```python
 python finetune.py
```
