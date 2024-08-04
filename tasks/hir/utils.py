import torch
import scipy.io as scio
import numpy as np
import random



def generate_masks(mask_path, size):
    mask256 = scio.loadmat(mask_path + '/mask' + str(size) + '.mat')
    mask256 = mask256['mask']
    # mask = np.transpose(mask, [2, 0, 1])
    mask = np.zeros((31, 286, 256))
    for i in range(31):
        mask[i, i:i+256, :]=mask256
    mask_s = np.sum(mask, axis=0)
    index = np.where(mask_s == 0)
    mask_s[index] = 1
    mask_s = mask_s.astype(np.uint8)
    mask = torch.from_numpy(mask)
    # mask = mask.float()
    # mask = mask.cuda()
    mask_s = torch.from_numpy(mask_s)
    # mask_s = mask_s.float()
    # mask_s = mask_s.cuda()
    return mask, mask_s

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def center_crop2(img, target_size):
    # img: [C,H,W]
    w, h = target_size
    _, wo, ho = img.shape
    return img[:, (wo-w)//2:(wo-w)//2+w, (ho-h)//2:(ho-h)//2+h]

def random_crop(img, target_size):
    # img: [C,W,H]
    w, h = target_size
    wo, ho, _ = img.shape
    _w = random.randint(0, wo - w)
    _h = random.randint(0, ho - h)
    return img[_w:_w+w, _h:_h+h, :]

def shift(inputs, step=1):
    batch,nC,row,col = inputs.shape
    output = torch.zeros((batch, nC, row+(nC-1)*step, col)).to(inputs.device)
    for i in range(nC):
        output[:,i,i*step:i*step+row,:] = inputs[:,i,:,:]
    return output

# def shift_back(inputs, step=1):
#     batch,nC,row,col = inputs.shape
#     for i in range(nC):
#         inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:],(-1)*step*i,dims=2)
#     output = inputs[:,:,0:row-step*(nC-1),:]
#     return output
def shift_back(inputs, step=1, flag=False):
    batch,nC,row,col = inputs.shape
    if flag:
        output = torch.zeros((batch, nC, row-30*step, col)).to(inputs.device)
    else:
        output = torch.zeros((batch, nC, row-(nC-1)*step, col)).to(inputs.device)
    for i in range(nC):
        output[:,i,:,:] = inputs[:,i,i*step:i*step+col,:]
    return output

def shift_back_numpy(inputs,step):
    [row,col,nC] = inputs.shape
    for i in range(nC):
        inputs[:,:,i] = np.roll(inputs[:,:,i],(-1)*step*i,axis=0)
    output = inputs[0:row-step*(nC-1),:,:]
    return output