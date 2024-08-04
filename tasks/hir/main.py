#!/usr/bin/env python3
import torch
import torch.utils.data.dataloader
from tensorboardX import SummaryWriter
from scipy.io import loadmat
from pathlib import Path

from env import HIREnv
from dataset import HIRDataset
from solver import create_solver_hir

from tfpnp.policy.sync_batchnorm import DataParallelWithCallback
from tfpnp.policy import create_policy_network
from tfpnp.pnp import create_denoiser
from tfpnp.trainer import MDDPGTrainer
from tfpnp.trainer.mddpg.critic import ResNet_wobn
from tfpnp.eval import Evaluator
from tfpnp.utils.noise import GaussianModelD
from tfpnp.utils.options import Options


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opt):
    data_dir = Path('./data/icvl')
    mask_dir = Path('data')
    log_dir = Path('log') / opt.exp
    mask_dir = mask_dir / 'mask'

    # sampling_masks = ['mask256', 'cs_mask_cassi']
    # sampling_masks = ['hir_mask256', 'hir_mask512']
    sampling_masks = ['mask256', 'mask512', 'mask1024']
    masks = [loadmat(mask_dir / f'{sampling_mask}.mat').get('mask') for sampling_mask in sampling_masks]

    
    base_dim = HIREnv.ob_base_dim
    actor = create_policy_network(opt, base_dim).to(device)  # policy network
    denoiser = create_denoiser(opt).to(device)
    solver = create_solver_hir(opt, denoiser, device).to(device)
    num_var = solver.num_var
    
    # ---------------------------------------------------------------------------- #
    #                                     Valid                                    #
    # ---------------------------------------------------------------------------- #
    writer = SummaryWriter(log_dir)
    

    val_root = data_dir
    val_dataset = HIRDataset(val_root, fns=None, masks=masks, num=512, target_size=(512,512), model='val')
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                               num_workers=0, pin_memory=True)
    val_name = 'hir_test'
    val_loaders = {val_name: val_loader}
    
    if torch.cuda.device_count() > 1:
        solver = DataParallelWithCallback(solver)

    eval_env = HIREnv(None, solver, max_episode_step=opt.max_episode_step, device=device)
    # evaluator = Evaluator(opt, eval_env, val_loaders, writer, device, 'results')
    evaluator = Evaluator(opt, eval_env, val_loaders, writer, device, 'presentation')
    
    if opt.eval:
        actor_ckpt = torch.load(opt.resume)
        actor.load_state_dict(actor_ckpt, False)
        evaluator.eval(actor, step=opt.resume_step)
        return
    
    # ---------------------------------------------------------------------------- #
    #                                     Train                                    #
    # ---------------------------------------------------------------------------- #
    
    # train_root = data_dir / 'single_test'
    train_root = data_dir / 'train/train'
    train_dataset = HIRDataset(train_root, fns=None, masks=masks, num=256, target_size=(256,256), model='train')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.env_batch, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    env = HIREnv(train_loader, solver, max_episode_step=opt.max_episode_step, device=device)

    def lr_scheduler(step):
        if step < 10000:
            return {'critic': 3e-4, 'actor': 1e-3}
        else:
            return {'critic': 1e-4, 'actor': 3e-4}

    critic = ResNet_wobn(base_dim+num_var, 18, 1).to(device)
    critic_target = ResNet_wobn(base_dim+num_var, 18, 1).to(device)

    trainer = MDDPGTrainer(opt, env, actor=actor,
                           critic=critic, critic_target=critic_target,
                           lr_scheduler=lr_scheduler, device=device,
                           evaluator=evaluator, writer=writer)
    trainer.load_model('./checkpoints/test')
    
    trainer.train()


if __name__ == "__main__":
    option = Options()
    opt = option.parse()
    main(opt)
