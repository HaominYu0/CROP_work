import os
import yaml
import shutil
import sys
import time
import warnings
import numpy as np
from random import sample
from sklearn import metrics
from datetime import datetime
from torch_sparse import SparseTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from dataset.data_pretrain_mask import get_train_val_test_loader
from model.dimenet_pretrain import DimeNetPlusPlusWrap
from model.tang_graph import GNN, GNNDecoder
from model.dimenet_pretrain import (
    get_pbc_distances, radius_graph_pbc, frac_to_cart_coords)
MAX_ATOMIC_NUM=95

OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

EPSILON = 1e-5

def min_distance_sqr_pbc(cart_coords1, cart_coords2, lengths, angles,
                         num_atoms, device, return_vector=False,
                         return_to_jimages=False):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    batch_size = len(num_atoms)

    # Get the positions for each atom
    pos1 = cart_coords1
    pos2 = cart_coords2

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(cart_coords2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # shape (num_atom_squared_sum, 3, 27)
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = torch.sum(atom_distance_vector ** 2, dim=1)

    min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)

    return_list = [min_atom_distance_sqr]

    if return_vector:
        min_indices = min_indices[:, None, None].repeat([1, 3, 1])

        min_atom_distance_vector = torch.gather(
            atom_distance_vector, 2, min_indices).squeeze(-1)

        return_list.append(min_atom_distance_vector)

    if return_to_jimages:
        to_jimages = unit_cell.T[min_indices].long()
        return_list.append(to_jimages)

    return return_list[0] if len(return_list) == 1 else return_list

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords

    return pos


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)

def cart_to_frac_coords(
    cart_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    return (frac_coords % 1.)




class pretrain_ENDE(nn.Module):
    def __init__(self):
        super(pretrain_ENDE, self).__init__()
        self.encoder = DimeNetPlusPlusWrap()
        self.latent_dim = 256
        self.hidden_dim = 256
        self.fc_num_layers = 2
        self.fc_graph_lyers = 3
        self.type_sigma_begin = 5.
        self.type_sigma_end=0.01
        self.num_noise_level = 50
        self.device =  'cuda'
        self.fc_lattice = build_mlp(self.latent_dim, self.hidden_dim, self.fc_num_layers, 6)

        self.fc_lattice_de = build_mlp(self.latent_dim, self.hidden_dim, self.fc_num_layers, 6)
        
        self.typeloss = nn.CrossEntropyLoss(reduction='none')

        
        self.emb_dim = 64
        self.JK = "last"
        self.gnn_type = "gin"
        self.num_layer = 5
        self.dropout_ratio=0
        NUM_NODE_ATTR = 119
        self.atom_pred_decoder = GNNDecoder(self.emb_dim, int(self.emb_dim*2), JK=self.JK, gnn_type=self.gnn_type)
        NUM_BOND_ATTR = 3
        self.bond_pred_decoder = GNNDecoder(self.emb_dim, NUM_BOND_ATTR, JK=self.JK, gnn_type=self.gnn_type)
        self.edge_layer = build_mlp(int(self.emb_dim*2), self.hidden_dim, self.fc_num_layers, 30)
        self.pred_node =  build_mlp(int(self.emb_dim*2), self.hidden_dim, self.fc_num_layers, NUM_NODE_ATTR)
        self.pred_lattice_en =  build_mlp(6, self.hidden_dim, self.fc_num_layers, 12)#build_mlp(6, 3, self.fc_num_layers, 3)
        self.pred_lattice_de =  build_mlp(12, self.hidden_dim, self.fc_num_layers, 6)
    
    def coord_loss(self, pred_cart_coord_diff, batch, masked_atom_indices1, mask_node):
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords[masked_atom_indices1], batch.lengths, batch.angles, mask_node)
        
        target_avg_coords = frac_to_cart_coords(
                            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        cart_avg_coords = scatter(target_avg_coords, batch.batch, dim=0, reduce='mean')
        cart_avg_coords = torch.repeat_interleave(cart_avg_coords, mask_node, dim=0)

        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, cart_avg_coords, batch.lengths, batch.angles, mask_node, self.device, return_vector=True)
        #import ipdb
        #ipdb.set_trace()
        #loss_per_atom = torch.sqrt(torch.sum(
           # (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1))
        #loss_per_atom = F.mse_loss(target_cart_coord_diff, pred_cart_coord_diff)
        #loss_per_atom = 0.5 * loss_per_atom **2
        loss_per_atom = F.l1_loss(target_cart_coord_diff,pred_cart_coord_diff, reduce=False).mean(dim=1)
        return scatter(loss_per_atom, batch.batch[masked_atom_indices1], reduce='mean').mean()
        
    
    def lattice_loss(self, pred_lengths_and_angles, batch):
        target_lengths_and_angles = batch.scaled_lattice
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)


    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()


    def type_loss(self, pred_atom_types, target_atom_types, batch):
        target_atom_types = target_atom_types
        loss = self.typeloss(pred_atom_types, target_atom_types)
        return scatter(loss, batch, reduce='mean').mean()
    
    def pred_edge_loss(self, edge_pred_attr, edge_gt_attr, batch):
        loss1 = F.cross_entropy(edge_pred_attr[:,:27], edge_gt_attr[:,0].long())
        loss2 =  F.cross_entropy(edge_pred_attr[:,27:29], edge_gt_attr[:,1].long())
        loss3 = F.l1_loss(edge_pred_attr[:,-1:], edge_gt_attr[:,-1:])
        return loss1+loss2+loss3



    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def predict_lattice(self, z, num_atoms,scaled_mean_std):
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        lengths_and_angles = pred_lengths_and_angles*scaled_mean_std.stds.cuda()+scaled_mean_std.means.cuda()
        pred_lengths = lengths_and_angles[:, :3]
        pred_angles = lengths_and_angles[:, 3:]
        pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_lattice(self, z, num_atoms,scaled_mean_std):
        pred_lengths_and_angles = self.fc_lattice_de(z)  # (N, 6)
        lengths_and_angles = pred_lengths_and_angles*scaled_mean_std.stds.cuda()+scaled_mean_std.means.cuda()
        pred_lengths = lengths_and_angles[:, :3]
        pred_angles = lengths_and_angles[:, 3:]
        pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        return pred_lengths_and_angles, pred_lengths, pred_angles


    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)
    
    def graph_loss(self, gt_edge, pre_edge, batch):
        row_gt, col_gt = gt_edge
        value = torch.ones(row_gt.size(0), device=row_gt.device)
        adj_gt = SparseTensor(row=col_gt, col=row_gt, value=value, sparse_sizes=(batch.num_nodes, batch.num_nodes))
        
        row_pre, col_pre = pre_edge
        value = torch.ones(row_pre.size(0), device=row_gt.device)
        adj_pre = SparseTensor(row=col_pre, col=row_pre, value=value, sparse_sizes=(batch.num_nodes, batch.num_nodes))
        return F.mse_loss(adj_gt.to_dense(), adj_pre.to_dense())
   
    def avg_coords(self, batch):
       target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
       cart_avg_coords = scatter(target_cart_coords, batch.batch, dim=0, reduce='mean')

       return avg_coords

    def forward(self, batch_gt):
        batch_gt = batch_gt.cuda()
        masked_atom_indices1 = batch_gt.masked_atom_indices1
        masked_atom_indices2 = batch_gt.masked_atom_indices2
        hidden_atom, hidden, gt_edge, edge_attr  = self.encoder(batch_gt)
        edge_index = gt_edge.long()
        idx,jdx = edge_index
        node_rep = hidden_atom
        node_rep_mask1 = node_rep.clone()
        node_rep_mask1[masked_atom_indices1] = 0
        edge_mask1 = torch.cat([node_rep_mask1[idx], node_rep_mask1[jdx]],-1)


        node_rep_mask2 = node_rep.clone()
        node_rep_mask2[masked_atom_indices2]=0
        edge_mask2 = torch.cat([node_rep_mask2[idx], node_rep_mask2[jdx]],-1)
        

        node_rep_mask1 = scatter(node_rep_mask1, batch_gt.batch, dim=0, reduce='mean')


        node_rep_mask2 = scatter(node_rep_mask2, batch_gt.batch, dim=0, reduce='mean')
        #        

        

        pred_node = self.atom_pred_decoder(node_rep, edge_index, edge_attr, masked_atom_indices1)
        pred_node2 = self.atom_pred_decoder(node_rep, edge_index, edge_attr, masked_atom_indices2)
        
        node_rep_mask1 = scatter(pred_node, batch_gt.batch, dim=0, reduce='mean')
        node_rep_mask2 = scatter(pred_node2, batch_gt.batch, dim=0, reduce='mean')

        pred_node = self.pred_node(pred_node)
        pred_node2 = self.pred_node(pred_node2)
        #import ipdb
        #ipdb.set_trace()





        cart_coords1 = self.bond_pred_decoder(node_rep, edge_index, edge_attr, masked_atom_indices1)
        cart_coords2 = self.bond_pred_decoder(node_rep, edge_index, edge_attr, masked_atom_indices2)
        edge_pred_attr1 = self.edge_layer(edge_mask1)
        edge_pred_attr2 = self.edge_layer(edge_mask2)
        
        edge_loss1 = self.pred_edge_loss(edge_pred_attr1,edge_attr, batch_gt)
        edge_loss2 = self.pred_edge_loss(edge_pred_attr2,edge_attr, batch_gt )
        egde_whole = (edge_loss1+edge_loss2)/2

        

        lattice_coord = batch_gt.scaled_lattice_tensor
        latent_pred =self.pred_lattice_de(self.pred_lattice_en(lattice_coord.view(lattice_coord.shape[0],-1)))
        loss_lattice = F.l1_loss(lattice_coord.view(lattice_coord.shape[0],-1),latent_pred)
        
        coord_loss1 = self.coord_loss(cart_coords1[masked_atom_indices1], batch_gt, masked_atom_indices1, batch_gt.mask_node1)
        coord_loss2 = self.coord_loss(cart_coords2[masked_atom_indices2], batch_gt, masked_atom_indices2, batch_gt.mask_node2)
        coord_loss2 = (coord_loss1 + coord_loss2)/2
        

        type_loss = self.type_loss(pred_node[masked_atom_indices1], batch_gt.atom_types[masked_atom_indices1], batch_gt.batch[masked_atom_indices1])
        type_loss2 = (type_loss+self.type_loss(pred_node2[masked_atom_indices2], batch_gt.atom_types[masked_atom_indices2], batch_gt.batch[masked_atom_indices2]))/2
        hidden1_loss = node_rep_mask1#torch.concat([node_rep_mask1, latent1_coord.view(latent1_coord.shape[0],-1)], dim=-1)
        hidden2_loss = node_rep_mask2#torch.concat([node_rep_mask2, latent2_coord.view(latent2_coord.shape[0],-1)], dim=-1)
        #a = 0
        return hidden1_loss, hidden2_loss, coord_loss2, type_loss2, egde_whole, loss_lattice
