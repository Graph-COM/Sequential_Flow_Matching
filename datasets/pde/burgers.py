# from .apps.burgers_h5py import Burgers
import torch
import torch.nn as nn

import scipy.io
import numpy as np
import h5py
import pdb
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from omegaconf import DictConfig

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

class BurgersDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        if split == "training":
            self.save_dir = Path(cfg.save_dir) / "train_data"
        elif split == 'finetune':
            self.save_dir = Path(cfg.save_dir) / "train_data"
        elif split in ["validation", "test"]:
            self.save_dir = Path(cfg.save_dir) / "test"
            #self.save_dir = Path(cfg.save_dir) / "train_data"
        else:
            raise ValueError(f"Invalid split: {split}")

        self.split = split

        self.rescaler = cfg.rescaler
        self.stack_u_and_f = cfg.stack_u_and_f 
        self.pad_for_2d_conv = cfg.pad_for_2d_conv 
        self.partially_observed_fill_zero_unobserved =  cfg.partially_observed_fill_zero_unobserved
        self.n_frames = cfg.episode_len + 1
        self.u_target = cfg.u_target
        self.load_all = True 
        if self.load_all:
            self.db = torch.load(self.save_dir)
            self.x = self._preprocess(self.db)
        else:
            raise NotImplementedError("Lazy loading not implemented")

    def _preprocess(self, db):
        """Preprocess the dataset according to cfg"""
        rescaler = self.cfg.get('rescaler')
        if rescaler is None:
            raise NotImplementedError("Should specify rescaler. If no rescaler is used, specify 1.")

        stack_u_and_f = self.cfg.stack_u_and_f
        pad_for_2d_conv = self.cfg.pad_for_2d_conv
        partially_observed_fill_zero_unobserved = self.cfg.partially_observed_fill_zero_unobserved

        if self.split == "training":
            # only use the first 1000 data for training
            #u = db['u'][:1000]
            #f = db['f'][:1000]
            # FIXME
            u = db['u'][:90000]
            f = db['f'][:90000]
        elif self.split in ["validation"] and self.u_target:
            u = db['u']
            # this is the original data split
            # For validation, take first 50 batches' time=0 as initial condition
            #u0 = u[:50, 0].unsqueeze(1)
            # Following CLDiffPhyCon, we take last 50 batches' states except time=0 as 
            # the rest of the trajectory as the target state.
            #u_target = u[-50:, 1:]  # shape: (50, n_frames-1, ...)
            #u = torch.cat([u0, u_target], dim=1)  # shape: (50, n_frames, ...)
            # f will not be used
            #f = torch.zeros_like(u)

            # this is "directly use original trajectory"
            f = db['f']

        elif self.split in ["validation"] and not self.u_target:
            u = db['u'][:100]
            f = db['f'][:100]
            #u = db['u'][90000:90000+1000]
            #f = db['f'][90000:90000+1000]
        elif self.split == 'finetune' and not self.u_target:
            n_finetune_data = 10000
            u = db['u'][90000:90000+n_finetune_data]
            f = db['f'][90000:90000+n_finetune_data]

        f = f[:, :(self.n_frames-1)]

        # Partial observation handling
        fill_zero_unobserved = partially_observed_fill_zero_unobserved
        if fill_zero_unobserved is not None:
            if fill_zero_unobserved == 'front_rear_quarter':
                u = u.squeeze()
                nx = u.shape[-1]
                u[..., nx // 4: (nx * 3) // 4] = 0
            else:
                raise ValueError('Unknown partially observed mode')

        # if stack_u_and_f:
        #     assert pad_for_2d_conv
        #     nt = f.size(-2)
        #     # padding f at the end
        #     f = nn.functional.pad(f, (0, 0, 0, self.n_frames - nt), 'constant', 0)
        #     u = nn.functional.pad(u, (0, 0, 0, (self.n_frames-1) - nt), 'constant', 0)
        #     u_target = u
        #     data = torch.stack((u, f, u_target), dim=1)
        # else:
        #     assert not pad_for_2d_conv
        #     data = torch.cat((u, f), dim=1)

        if self.u_target:
            assert pad_for_2d_conv
            nt = f.size(-2)
            # padding f at the end
            f = nn.functional.pad(f, (0, 0, 0, self.n_frames - nt), 'constant', 0)
            u = nn.functional.pad(u, (0, 0, 0, (self.n_frames-1) - nt), 'constant', 0)
            u_target = u
            data = torch.stack((u, f, u_target), dim=1)
        else:
            assert pad_for_2d_conv
            nt = f.size(-2)
            f = nn.functional.pad(f, (0, 0, 0, self.n_frames - nt), 'constant', 0)
            u = nn.functional.pad(u, (0, 0, 0, (self.n_frames-1) - nt), 'constant', 0)
            data = torch.stack((u, f), dim=1)

        data = data / rescaler
        return data

    def __len__(self):
        if self.load_all:
            return self.x.size(0)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.load_all:
            return self.x[idx]
        else:
            raise NotImplementedError

    def get(self, idx):
        return self.__getitem__(idx)
        
def get_burgers_preprocess(
    rescaler=None, 
    stack_u_and_f=False, 
    pad_for_2d_conv=False, 
    partially_observed_fill_zero_unobserved=None, 
):
    if rescaler is None:
        raise NotImplementedError('Should specify rescaler. If no rescaler is not used, specify 1.')
    
    def preprocess(db):
        '''We are only returning f and u for now, in the shape of 
        (u0, u1, ..., f0, f1, ...)
        '''
        
        u = db['u']
        f = db['f']
        f = f[:,:15]

        fill_zero_unobserved = partially_observed_fill_zero_unobserved
        if fill_zero_unobserved is not None:
            if fill_zero_unobserved == 'front_rear_quarter':
                u = u.squeeze()
                nx = u.shape[-1]
                u[..., nx // 4: (nx * 3) // 4] = 0
            else:
                raise ValueError('Unknown partially observed mode')

        if stack_u_and_f:
            assert pad_for_2d_conv
            nt = f.size(-2)
            f = nn.functional.pad(f, (0, 0, 0, 16 - nt), 'constant', 0)
            u = nn.functional.pad(u, (0, 0, 0, 15 - nt), 'constant', 0)
            u_target = u  
            data = torch.stack((u, f, u_target), dim=1) 
        else:
            assert not pad_for_2d_conv
            data = torch.cat((u, f), dim=1)
    
        data = data / rescaler
        return data

    return preprocess



class LegacyBurgersDataset(Dataset):
    def __init__(
        self, 
        fname, 
        preprocess=get_burgers_preprocess('all'),  
        load_all=True
    ):
        '''
        Arguments:

        '''
        self.load_all = load_all
        if load_all:
            self.db = torch.load(fname)
            self.x = preprocess(self.db)
        else:
            raise NotImplementedError

    def __len__(self):
        if self.load_all:
            return self.x.size(0)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.load_all:
            return self.x[idx]
        else:
            raise NotImplementedError

    def get(self, idx):
        return self.__getitem__(idx)
    
    def len(self):
        return self.__len__()


if __name__=='__main__':

    tmp = LegacyBurgersDataset('data/pde/train_data', 
        preprocess=get_burgers_preprocess(
            rescaler=10, 
            stack_u_and_f=True, 
            pad_for_2d_conv=True, 
            partially_observed_fill_zero_unobserved = None 
        ) 
    )

    import pdb; pdb.set_trace()
