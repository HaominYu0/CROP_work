from __future__ import print_function, division
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
import csv
import functools
import  json
#import  you
import  random
import warnings
import math
import  numpy  as  np
import torch
import scipy.sparse as ss 
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os
from dataset.batch import BatchMasking
from p_tqdm import p_umap
import pandas as pd
from torch_geometric.data import Data
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset
#from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.operations import SymmOp
from pymatgen.transformations.transformation_abc import AbstractTransformation
from .utils import StandardScalerTorch

def collate_pool(batches):
    batches = [x for x in batches]
    batches = BatchMasking.from_data_list(batches)
    return batches

def get_train_val_test_loader(dataset, collate_fn,
                              batch_size=64, val_ratio=0.1, num_workers=1, 
                              pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.
    !!! The dataset needs to be shuffled before using the function !!!
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool
    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    train_ratio = 1 - val_ratio
    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    print(total_size, train_size, valid_size)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[train_size:])
    
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers, drop_last=True,
                              collate_fn=collate_fn, 
                              pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers, drop_last=True,
                            collate_fn=collate_fn, 
                            pin_memory=True)
    return train_loader, val_loader




class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:
    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...
    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.
    atom_init.json: a JSON file that stores the initialization vector for each
    element.
    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.
    Parameters
    ----------
    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset
    Returns
    -------
    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self . root_dir  =  root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        # id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        # assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        # with open(id_prop_file) as f:
        #     reader = csv.reader(f)
        #     self.id_prop_data = [row for row in reader]
        target = 'id_prop.csv'
        id_prop_file = os.path.join(self.root_dir, target)
        reader = pd.read_csv(id_prop_file)
        #import ipdb
        #ipdb.set_trace()
        self.id_prop_data = np.array(reader[["_oqmd_entry_id"]]).tolist()
        cif_fns  = []
        #df_csv = pd.read_csv(root_dir + target)
        material_id_list = self.id_prop_data#df_csv['_oqmd_entry_id'].to_list()
        #for material_id in material_id_list:
        #    cif_fns.append(root_dir+str(material_id[0])+'.cif')
        for subdir, dirs, files in os.walk(root_dir):
            for  fn  in  files :
                if fn.endswith('.cif'):
                    if fn != '561810.cif':
                        cif_fns.append(os.path.join(subdir, fn))
        self.cif_data = cif_fns
        
        self.cached_data = self.preprocess(self.id_prop_data, self.cif_data)
        self.add_scaled_lattice_prop(self.cached_data)
        
        lattice_scaler = self.get_scaler_from_data_list(
                            self.cached_data,
                            'scaled_lattice')
        self.lattice_scaler = lattice_scaler
        #self.MAX_NUM = MAX_NUM
        patch_list = []
        random.seed(random_seed)
        # random.shuffle(self.id_prop_data)
        random.shuffle(self.cif_data)
        # atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        atom_init_file = os.path.join('dataset/atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
    
    def  get_scaler_from_data_list(self,data_list, key):
        targets = torch.tensor([d[key] for d in data_list])
        scaler = StandardScalerTorch()
        scaler.fit(targets)
        return scaler

    def  get_num_from_data_list(self,data_list, key):
        targets = torch.tensor([d[key][-1] for d in data_list])
        targets_max = torch.max(targets)
        return targets_max


    def preprocess(self, id_prop_data, cif_data):
        

        def process_one(id_data, cif_fn):
            #cif_id = id_data
            cif_id = cif_fn.split('/')[-1].replace('.cif', '') 
            crys = Structure.from_file(cif_fn)
            
            niggli=True
            if niggli:
                crys = crys.get_reduced_structure()

            crys =Structure(
                    lattice=Lattice.from_parameters(*crys.lattice.parameters),
                    species=crys.species,
                    coords=crys.frac_coords,
                    coords_are_cartesian=False)
            graph_arrays = self.crys_structure(crys)
            
            #crys_change = crys.copy()
            #crys_change1 = crys.copy()

            #crys_change.perturb(distance = 0.05, min_distance = 0.0)
            #crys_change1.perturb(distance = 0.02, min_distance = 0.0)

            #graph_change = self.crys_structure(crys_change)
            #graph_change2 = self.crys_structure(crys_change1)
            result_dict = {
                    'mp_id':cif_id, 
                    'cif': crys,
                    'graph_arrays':graph_arrays,
                    #'graph_change':graph_change,
                    #'graph_change2':graph_change2
                    }
            return result_dict
        
        #unordered_results = p_umap(
         #       process_one, 
        #        [id_prop_data[idx] for idx in range(len(id_prop_data)) ],
        #        [cif_data[idx] for idx in range(len(cif_data))],
       #         num_cpus = 60
       #         )
        len_id = len(id_prop_data)
        unordered_results = []
        for i in range(1000):
             id_prop_data_tmp = id_prop_data[int(len_id*i*0.001):int(len_id*(i+1)*0.001)]
             cif_data_tmp = cif_data[int(len_id*i*0.001):int(len_id*(i+1)*0.001)]
             unordered_results.extend(p_umap(process_one,[id_prop_data_tmp[idx] for idx in range(len(id_prop_data_tmp)) ],[cif_data_tmp[idx] for idx in range(len(cif_data_tmp))],))

        


        mpid_to_results = {result['mp_id']: result for result in unordered_results}
        ordered_results = [mpid_to_results[cif_data[idx].split('/')[-1].replace('.cif', '')]
                       for idx in range(len(cif_data))]

        #ordered_results = process_one(id_prop_data[0], cif_data[0])
        return ordered_results

    def add_scaled_lattice_prop(self, data_list):
       for dict in data_list:
            graph_arrays = dict['graph_arrays']
            # the indexes are brittle if more objects are returned
            lengths = graph_arrays[2]
            angles = graph_arrays[3]
            num_atoms = graph_arrays[-1]
            assert lengths.shape[0] == angles.shape[0] == 3
            assert isinstance(num_atoms, int)

            #lengths = lengths / float(num_atoms)**(1/3)

            dict['scaled_lattice'] = np.concatenate([lengths, angles])
    

    def crys_structure(self, crys):
            #CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=0, porous_adjustment=False)
            #try:
            #    crystal_graph = StructureGraph.with_local_env_strategy(crys, CrystalNN)
            #except (RuntimeError, TypeError, NameError, ValueError):
            #    print("crystal_error")
            #    crys = Structure.from_file('/data/cs.aau.dk/haominyu/cdvae/Dataset/MP_DATA_post/mp-1023940.cif')
            #    crystal_graph = StructureGraph.with_local_env_strategy(crys, CrystalNN)
            frac_coords = crys.frac_coords
            atom_types = crys.atomic_numbers

            lattice_parameters = crys.lattice.parameters
            lengths = lattice_parameters[:3]
            angles = lattice_parameters[3:]
            edge_indices, to_jimages = [], []
            
            edge_attr = torch.zeros((len(edge_indices), 2), dtype=torch.long)

            atom_types = np.array(atom_types)
            num_atoms = atom_types.shape[0]
            indices_whole = list(range(num_atoms))
            mask_num = int(max([1, math.floor(0.50*num_atoms)]))
            
            mask_node1 = mask_num
            mask_node2 = num_atoms - mask_num

            #if mask_node1 == 0:


            atom_type_mask1 = np.random.choice(num_atoms, mask_num, replace=False)
            mask_node_labels_list1 = []
            for atom_idx in atom_type_mask1:
                mask_node_labels_list1.append(atom_types[atom_idx])
            mask_node_labels_list1 = np.array(mask_node_labels_list1)
            atom_type_mask2 = np.array([i for i in indices_whole if i not in atom_type_mask1])
            
            if mask_node1 == 0:
                atom_type_mask1 = np.random.choice(1, mask_num, replace=False)
                atom_type_mask2 = np.random.choice(1, mask_num, replace=False)
                mask_node1 = 1
                mask_node2 = 1
            elif mask_node2 == 0:
                atom_type_mask1 = np.random.choice(1, mask_num, replace=False)
                atom_type_mask2 = np.random.choice(1, mask_num, replace=False)
                mask_node1 = 1
                mask_node2 = 1
            mask_node_labels_list2 = []
            for atom_idx in atom_type_mask2:
                mask_node_labels_list2.append(atom_types[atom_idx])
            mask_node_labels_list2 = np.array(mask_node_labels_list2)


             


            
            #for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            #    edge_indices.append([j, i])
            #    to_jimages.append(to_jimage)
            #    edge_indices.append([i, j])
            #    to_jimages.append(tuple(-tj for tj in to_jimage))

            edge_indices.append([0, 0])
            to_jimages.append([0,0])
            edge_attr = torch.zeros((len(edge_indices), 2), dtype=torch.long)
             
            #import ipdb
            #ipdb.set_trace()
            lattice_mask1 = np.array([1,0])
            lattice_mask2 = np.array([0,1])

            lengths, angles = np.array(lengths), np.array(angles)
            #scaled_lengths = np.array(lengths)/float(num_atoms)**(1/3)
            scaled_lattice = np.concatenate([lengths, angles]) 


            edge_indices = np.array(edge_indices)
            to_jimages = np.array(to_jimages)

            return frac_coords, atom_types, lengths,  angles, edge_indices, to_jimages, scaled_lattice, atom_type_mask1, atom_type_mask2, mask_node_labels_list1, mask_node_labels_list2, edge_attr, lattice_mask1, lattice_mask2, mask_node1, mask_node2, num_atoms#data

    
    def graph_file(self, graph_arrays, cif_id):
        (frac_coords, atom_types, lengths,  angles, edge_indices, to_jimages,scaled_lattice,  atom_type_mask1, atom_type_mask2,  mask_node_labels_list1, mask_node_labels_list2, edge_attr, lattice_mask1, lattice_mask2, mask_node1, mask_node2, num_atoms) = graph_arrays
       
        
        lattice_tensor = torch.concat([torch.Tensor(lengths).view(1, 1,-1), torch.Tensor(angles).view(1, 1,-1)],1)
        scaled_lattice = torch.Tensor(scaled_lattice)#-self.lattice_scaler.means)/self.lattice_scaler.stds
        scaled_lattice_tensor = torch.Tensor(scaled_lattice.reshape(1,2,3))
        lattice_mask1 = torch.Tensor(np.expand_dims(lattice_mask1, -1).repeat(3, axis=-1)).view(1,-1,3)
        lattice_mask2 = torch.Tensor(np.expand_dims(lattice_mask2, -1).repeat(3, axis=-1)).view(1,-1,3)
        data = Data(
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                edge_attr = torch.Tensor(edge_attr),
                mask_node_labels_list1 = torch.LongTensor(mask_node_labels_list1),
                mask_node_labels_list2 = torch.LongTensor(mask_node_labels_list2),
                masked_atom_indices1 = torch.LongTensor(atom_type_mask1),
                masked_atom_indices2  = torch.LongTensor(atom_type_mask2),
                mask_node1 = mask_node1,
                mask_node2 = mask_node2,
                #frac_tensor = frac_tensor,
                #atom_tensor = atom_tensor,
                #atom_type_mask1_tensor = atom_type_mask1_tensor, 
                #atom_type_mask2_tensor = atom_type_mask2_tensor,
                #coord_type_mask1_tensor = coord_type_mask1_tensor,
                #coord_type_mask2_tensor = coord_type_mask2_tensor,
                lattice_tensor  = lattice_tensor,
                scaled_lattice_tensor = scaled_lattice_tensor,
                lattice_mask1 = lattice_mask1,
                lattice_mask2 = lattice_mask2,
                #mask_matrix = torch.Tensor(mask_matrix).view(1,self.MAX_NUM,self.MAX_NUM), 
                #graph_matrix = torch.Tensor(graph_matrix).view(1,self.MAX_NUM,self.MAX_NUM),
                #graph_mask1 = torch.Tensor(graph_mask1).view(1,self.MAX_NUM,self.MAX_NUM), 
                #graph_mask2 = torch.Tensor(graph_mask2).view(1,self.MAX_NUM,self.MAX_NUM),
                #graph_matrix1 = torch.Tensor(graph_matrix1).view(1,self.MAX_NUM,self.MAX_NUM),
                #graph_matrix2 = torch.Tensor(graph_matrix2).view(1,self.MAX_NUM,self.MAX_NUM), 


                lengths=torch.Tensor(lengths).view(1, -1),
                scaled_lattice = torch.Tensor(scaled_lattice).view(1, -1),
                #scaled_mean_std = self.lattice_scaler,
                angles=torch.Tensor(angles).view(1, -1),
                edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),
                to_jimages=torch.LongTensor(to_jimages),
                num_atoms=num_atoms,
                num_bonds=edge_indices.shape[0],
                num_nodes=num_atoms,
                cif_id = cif_id)
        return data



    def __len__(self):
        # return len(self.id_prop_data)
        return len(self.cif_data)

    #@functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
            data_dict = self.cached_data[idx]
            
            crys = data_dict['cif']
            mp_id = data_dict['mp_id']
            #target = data_dict['target']

            data = self.graph_file(data_dict['graph_arrays'], mp_id)
            #data1 = self.graph_file(data_dict['graph_change'], mp_id, target)
            #data2 = self.graph_file(data_dict['graph_change2'], mp_id, target)
            return data#, data1, data2

