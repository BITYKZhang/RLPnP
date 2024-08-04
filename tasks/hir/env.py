
import torch
from tfpnp.data.batch import Batch
from tfpnp.env import PnPEnv
from tfpnp.utils.transforms import complex2channel, complex2real
from torchvision.transforms import Resize
from utils import shift, shift_back

class HIREnv(PnPEnv):
    # class attribute: the dimension of ob (exclude solver variable)
    ob_base_dim = 123  
    def __init__(self, data_loader, solver, max_episode_step, device):
        super().__init__(data_loader, solver, max_episode_step, device)
        self.resize = Resize((128,128))
    
    def get_policy_ob(self, ob):
        # print('policy----------------')
        # print(ob.y0.shape)
        # print(ob.variables.shape)
        # print(ob.mask.shape)
        # print(ob.T.shape)

        ans = torch.cat([
            ob.variables,
            ob.y0,
            ob.mask,
            ob.T,
        ], 1)

        return ans
    
    def get_eval_ob(self, ob):
        return self.get_policy_ob(ob)
    
    def _get_attribute(self, ob, key):
        if key == 'gt':
            return ob.gt
        elif key == 'output':
            return self.solver.get_output(ob.variables)
        elif key == 'input':
            return ob.y0.squeeze(1)
        elif key == 'solver_input':
            return (ob.variables, (ob.y0.squeeze(1), ob.mask.bool()))
        else:
            raise NotImplementedError('key is not supported, ' + str(key))
        
    def _build_next_ob(self, ob, solver_state):
        return Batch(gt=ob.gt,
                     y0=ob.y0,
                     variables=solver_state,
                     mask=ob.mask,
                     T=ob.T)
    
    def _observation(self):
        idx_left = self.idx_left
        return Batch(gt=self.state['gt'][idx_left, ...],
                     y0=self.state['y0'][idx_left, ...].unsqueeze(1),
                     variables=self.state['solver'][idx_left, ...],
                     mask=self.state['mask'][idx_left, ...].float(),
                     T=self.state['T'][idx_left, ...])
