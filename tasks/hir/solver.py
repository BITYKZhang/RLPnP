import torch
import numpy as np
from tfpnp.env.base import torch_psnr

from tfpnp.pnp.solver.base import ADMMSolver
from tfpnp.utils import transforms
from tfpnp.utils.metric import mpsnr_max
from utils import shift, shift_back


# decorator
def complex2real_decorator(func):
    def real_func(*args, **kwargs):
        return transforms.complex2real(func(*args, **kwargs))
    return real_func


class HIRMixin:
    # @complex2real_decorator
    # def get_output(self, state):
    #     return super().get_output(state)

    def filter_aux_inputs(self, state):
        return (state['y0'], state['mask'])


class ADMMSolver_HIR(HIRMixin, ADMMSolver):
    #TODO warning: HIRMixin must be put behind the ADMMSolver class
    # def __init__(self, denoiser, target):
    #     super().__init__(denoiser)
    #     self.gt = target[0]
    def __init__(self, denoiser, device):
        super().__init__(denoiser)
        self.device = device

    def acquireTarget(self, target):
        self.gt = target.detach().cpu().numpy()[0]

    def get_output(self, state):
        # x's shape [B,1,W,H]
        x, _, _ = torch.split(state, state.shape[1] // 3, dim=1)
        x = shift_back(x, 1)
        return x

    def denoise(self, value, sigma):
        output = []
        for band in value.split(1, 1):
            band = self.prox_mapping(band, sigma)
            output.append(band)
        return torch.cat(output, dim=1)

    # def comparePara(self, x):
    #     x = x.squeeze(0)
    #     # print(x[0].shape)
    #     # print(self.gt[0].shape)
    #     psnr = torch_psnr(x, self.gt)
    #     # print(psnr.shape)
    #     return torch.mean(psnr)

    def forward(self, inputs, parameters, iter_num=None):    
        
        variables, (y0, mask) = inputs
        sigmas, rhos = parameters

        # print("----------------------")
        # print(variables.shape)
        # print(y0.shape)
        # print(mask.shape)
        # print(rhos.shape)
        # print(sigmas.shape)

        # x,v,u [B,C,W,H]
        x, v, u = torch.split(variables, variables.shape[1]//3, dim=1)
        # print('----------', 0, '----------', mpsnr_max(shift_back(x, 1).detach().cpu().numpy()[0], self.gt))

        # infer iter_num from provided hyperparameters
        if iter_num is None:
            iter_num = rhos.shape[-1]

        A = lambda x : torch.sum(x*mask, dim=1)
        At = lambda x : x.unsqueeze(dim=1)*mask

        phi = torch.sum(mask**2, dim=1)
        
        for i in range(iter_num):
            # x step
            tau = rhos[:,i].float().unsqueeze(-1).unsqueeze(-1)
            # tau = rhos.float().unsqueeze(-1).unsqueeze(-1)
            xtilde = v - u
            rhs = At((y0-A(xtilde))/(phi+tau+1e-8))
            x = xtilde + rhs

            # z step
            vtilde = x + u
            vtilde = shift_back(vtilde, 1)
            with torch.no_grad():
                # print(vtilde.shape)
                # print(sigmas[:,i].shape)
                v = self.prox_mapping(vtilde, sigmas[:,i])
                v = shift(v, 1)


            # u step
            u = u + x - v

            # self.comparePara(tau, sigmas[:,i], x)
            # print('----------', i, '----------', mpsnr_max(shift_back(x, 1).detach().cpu().numpy()[0], self.gt))
        
        return torch.cat((x, v, u), dim=1)



_solver_map = {
    'admm': ADMMSolver_HIR
}

def create_solver_hir(opt, denoiser, device):
    print(f'[i] use solver: {opt.solver}')
    
    if opt.solver in _solver_map:
        solver = _solver_map[opt.solver](denoiser, device)
    else:
        raise NotImplementedError

    return solver
