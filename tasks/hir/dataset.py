import os
import random
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import torch 
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat
import h5py

from utils import A, At, center_crop2, shift, shift_back, generate_masks, random_crop

from tfpnp.data.util import center_crop, scale_height, scale_width, data_augment
from tfpnp.utils import transforms
from tfpnp.utils.transforms import complex2real
from tfpnp.utils.metric import mpsnr_max


class HIRDataset(Dataset):
    def __init__(self, datadir, fns, masks, noise_model=None, size=None, target_size=None, repeat=1, augment=False, num=256, model='train'):
        super().__init__()
        self.datadir = datadir
        # self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]      
        self.fns = fns or [im for im in os.listdir(self.datadir)]      
        self.fns = sorted(self.fns)        
        self.masks = masks
        self.noise_model = noise_model
        self.size = size
        self.repeat = repeat
        self.target_size = target_size
        self.augment = augment
        self.num = num
        self.model = model

    
    # SD
    def __getitem__(self, index):
        if self.num == 256:
            mask_s = self.masks[0].astype('float32')
        elif self.num == 512:
            mask_s = self.masks[1].astype('float32')
        elif self.num == 1024:
            mask_s = self.masks[2].astype('float32')
        mask_s = torch.from_numpy(mask_s)      # [W,H]
        B = 31
        mask = torch.zeros((mask_s.shape[0]+B-1, mask_s.shape[1], B))   # [W+30,H,31]
        for i in range(B):
            mask[i:i+mask_s.shape[0],:,i] = mask_s
        mask = mask.permute(2,0,1)              # [31,W+30,H]

        index = index % len(self.fns)
        imgpath = os.path.join(self.datadir, self.fns[index])
        # print(index, imgpath)
        # return
        target = loadmat(imgpath).get('gt')
        # target = h5py.File(imgpath, 'r').get('rad')
        # target = h5py.File(imgpath, 'r').get('hyperImage')
        # print(target.shape)

        if self.augment:
            target = data_augment(target)
        
        if self.model == 'train':
            if self.target_size is not None:
                # mask = center_crop(mask.permute(1,2,0), self.target_size).permute(2,0,1)
                # target = random_crop(target, self.target_size)
                w, h = self.target_size
                wo, ho, _ = target.shape
                _w = random.randint(0, wo - w)
                _h = random.randint(0, ho - h)
                target = target[_w:_w+w, _h:_h+h, :]
        elif self.model == 'val':
            if self.target_size is not None:
                # mask = center_crop(mask.permute(1,2,0), self.target_size).permute(2,0,1)
                target = center_crop(target, self.target_size)

        
        target = torch.from_numpy(target).float().permute(2,0,1)  # [31,W,H]
        truth = shift(target.unsqueeze(0)).squeeze()   # [31,W+30,H]
        y0 = torch.sum(truth*mask,dim=0)  # [W+30,H]

        At = lambda x: torch.unsqueeze(x, dim=0)*mask
        x = At(y0)                         # [31,W+30,H]

        n, _ = imgpath.split('/')[-1].split('.')
        # print(n)

        if self.model == 'train':
            d = loadmat('/data/results/reconstruction/CASSI/ICVL/TV/train/' + n +'.mat')['recon'].astype(np.float32).transpose(2,0,1)
            g = torch.from_numpy(d)
            g = g[:, _w:_w+w, _h:_h+h]
        elif self.model == 'val':
            d = h5py.File('/data/results/reconstruction/CASSI/ICVL/TV/test/' + n +'.h5', 'r')['recon']
            g = np.array(d).astype(np.float32)
            g = torch.from_numpy(g)
            g = center_crop(g.permute(1,2,0), self.target_size).permute(2,0,1)

        g = shift(g.unsqueeze(0), 1).squeeze()   # [31,W+30,H]

        
        dic = {'y0': y0, 'x0': g, 'gt': target, 
               'mask': mask, 'output': shift_back(x.unsqueeze(0), 1).squeeze(), 'input': x}

        # print('dataset----------------')
        # print(y0.shape)
        # print(target.shape)
        # print(mask.shape)
        # print(x.shape)
        
        return dic

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size
