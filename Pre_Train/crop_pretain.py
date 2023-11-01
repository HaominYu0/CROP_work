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
import dill
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from dataset.data_pretrain_mask import CIFData
from dataset.data_pretrain_mask import get_train_val_test_loader,collate_pool
from model.dimenet_pretrain import DimeNetPlusPlusWrap
from model.en_de_dimenet import pretrain_ENDE
from CT_transfer.loss.barlow_twins import BarlowTwinsLoss
from torch.nn.parallel import DistributedDataParallel as DDP
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

def _check_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

class CrystalContrastive(object):
    def __init__(self, config, loss_weight):
        self.config = config
        self.cost_coord=0.5
        self.cost_type=10.
        self.cost_lattice = 10.
        self.beta = 10
        self.loss_weight = loss_weight
        self.cost_composition=1.
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = "OQMD_weight_"+str(loss_weight)
        log_dir = os.path.join('runs_crop', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        #self.criterion =  NTXentLoss(self.device, **config['loss1'])

        
        data_loader_root = 'Dataset/'
        target_dataset = 'OQMD'
        _check_file(data_loader_root)
        data_loader_path = data_loader_root+target_dataset
        if os.path.exists(data_loader_path):
             with open(data_loader_path+'/train.pkl','rb') as f:
                   self.train_loader = dill.load(f)
             with open(data_loader_path+'/val.pkl','rb') as f:
                   self.valid_loader = dill.load(f)

        else:
            self.dataset = CIFData(**self.config['dataset'])
            collate_fn = collate_pool
            self.train_loader, self.valid_loader = get_train_val_test_loader(
                dataset=self.dataset,
                collate_fn=collate_pool,
                pin_memory=self.config['cuda'],
                batch_size=self.config['batch_size'], 
                **self.config['dataloader']
            )
            _check_file(data_loader_path)
            with open(data_loader_path+'/train.pkl','wb') as f:
                  dill.dump(self.train_loader, f)
            with open(data_loader_path+'/val.pkl','wb') as f:
                  dill.dump(self.valid_loader, f)




        self.criterion = BarlowTwinsLoss(self.device, **config['loss'])

    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print("GPU available")
        else:
            print("GPU not available")
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'#torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
            #torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)

        return device

    def _step(self, model, data):
        # get the representations and the projections
        #zis, coord_loss, type_loss, kld_loss, compos_loss, lattice  = model(data_i)  # [N,C]
       # zis,  zjs, atom_loss1, atom_loss2, coord_loss1, coord_loss2, graph_loss1,graph_loss2 
        zis, zjs, coord_loss2, type_loss2, edge_loss, loss_lattice =  model(data)
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        #loss_com = atom_loss1+atom_loss2+coord_loss1+coord_loss2
        loss_ba = self.criterion(zis, zjs)
        #self.loss_weight = 0.1*self.loss_weight
        loss = 0.1*self.loss_weight*(coord_loss2+ type_loss2+loss_lattice)+(1-0.1*self.loss_weight)*(loss_ba+edge_loss)# + lossi_whole + lossj_whole
        #self.criterion = nn.MSELoss()
        #import ipdb
        #ipdb.set_trace()
        #loss = self.criterion(zis, data_i.target)
        #print(zis)
        #print('loss:{:.5f}, loss_com:{:.5f}, loss_ba:{:.5f}, '.format( loss.item(), loss_com.item(), loss_ba.item() ))
        print('atom1:{:.5f},  coord1:{:.5f},  lattice:{:.5f} loss_ba:{:.5f}'.format(type_loss2.item(), coord_loss2.item(), loss_lattice.item(), loss_ba.item() ))
        return loss

    def train(self):

        model = pretrain_ENDE()#DimeNetPlusPlusWrap()
        model = self._load_pre_trained_weights(model)
        if self.config['cuda'] and torch.cuda.device_count()>1:
            #model = nn.DataParallel(model, device_ids = [0,1,2])
            model = model.to(self.device)
        elif self.config['cuda']:
            model = model.to(self.device)
    
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)

        if self.config['optim']['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), self.config['optim']['lr'],
                                momentum=self.config['optim']['momentum'],
                                weight_decay=eval(self.config['optim']['weight_decay']))
        elif self.config['optim']['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), self.config['optim']['lr'],
                                weight_decay=eval(self.config['optim']['weight_decay']))
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')        
        
        scheduler = CosineAnnealingLR(optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for bn, (input_gt) in enumerate(self.train_loader):
                loss = self._step(model, input_gt)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('TRAIN INFO: epoch:{} ({}/{}) iter:{} loss:{:.5f}'.format(epoch_counter, bn + 1, len(self.train_loader), n_iter, loss.item()))
                n_iter += 1

            torch.cuda.empty_cache()
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, self.valid_loader)
                print('Validation', valid_loss)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if epoch_counter  == 15 or epoch_counter  == 50 or epoch_counter  == 100 or epoch_counter  == 200:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(epoch_counter)))
            
            # warmup for the first 5 epochs
            #if epoch_counter >= 5:
            #    scheduler.step()
    
    def _load_pre_trained_weights(self, model):
        #print("Here")
        try:
            checkpoints_folder = os.path.join('./runs_crop', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model_5.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
            #print("loaded")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        #print("2st",os.system('free -h'))
        with torch.no_grad():
            model.eval()

            loss_total = 0.0
            total_num = 0
            for input_gt in valid_loader:
                loss = self._step(model, input_gt)
                loss_total += loss.item() * len(input_gt.cif_id)
                total_num += len(input_gt.cif_id)
            loss_total /= total_num
        #print("4th",os.system('free -h'))
        torch.cuda.empty_cache()
        model.train()
        return loss_total


if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    for loss_weight in [5]:
        crys_contrast = CrystalContrastive(config, loss_weight)
        crys_contrast.train()
