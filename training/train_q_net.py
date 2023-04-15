from argparse import ArgumentParser
import os
from comet_ml import Experiment

import d4rl
import gym
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.dqn import DDQN

class QLearningDataset(Dataset):
    def __init__(
        self, dataset_dir, filename, train_or_test="train", test_prop=0.1, sample_z=False
    ):
        # just load it all into RAM
        self.state_all = np.load(os.path.join(dataset_dir, filename + "_states.npy"), allow_pickle=True)
        self.latent_all = np.load(os.path.join(dataset_dir, filename + "_latents.npy"), allow_pickle=True)
        self.sT_all = np.load(os.path.join(dataset_dir, filename + "_sT.npy"), allow_pickle=True)
        self.rewards_all = (4*np.load(os.path.join(dataset_dir, filename + "_rewards.npy"), allow_pickle=True) - 30*4*0.5)/10 #zero-centering
        self.sample_z = sample_z
        if sample_z:
            self.latent_all_std = np.load(os.path.join(dataset_dir, filename + "_latents_std.npy"), allow_pickle=True)
        
        n_train = int(self.state_all.shape[0] * (1 - test_prop))
        if train_or_test == "train":
            self.state_all = self.state_all[:n_train]
            self.latent_all = self.latent_all[:n_train]
            self.sT_all = self.sT_all[:n_train]
            self.rewards_all = self.rewards_all[:n_train]
        elif train_or_test == "test":
            self.state_all = self.state_all[n_train:]
            self.latent_all = self.latent_all[n_train:]
            self.sT_all = self.sT_all[n_train:]
            self.rewards_all = self.rewards_all[n_train:]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.state_all.shape[0]

    def __getitem__(self, index):
        state = self.state_all[index]
        latent = self.latent_all[index]
        if self.sample_z:
            latent_std = self.latent_all_std[index]
            latent = np.random.normal(latent,latent_std)
        sT = self.sT_all[index]
        reward = self.rewards_all[index]

        return (state, latent, sT, reward)


def train(args):
    env = gym.make(args.env)
    dataset = env.get_dataset()
    state_dim = dataset['observations'].shape[1]
    a_dim = dataset['actions'].shape[1]

    # get datasets set up
    torch_data_train = QLearningDataset(
        args.dataset_dir, args.skill_model_filename[:-4], train_or_test="train", test_prop=args.test_split, sample_z=args.sample_z
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    torch_data_test = QLearningDataset(
        args.dataset_dir, args.skill_model_filename[:-4], train_or_test="test", test_prop=args.test_split, sample_z=args.sample_z
    )
    dataload_test = DataLoader(
        torch_data_test, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    x_shape = torch_data_train.state_all.shape[1]
    y_dim = torch_data_train.latent_all.shape[1]

    # create model
    diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt')).to(args.device)
    model = Model_Cond_Diffusion(
        diffusion_nn_model,
        betas=(1e-4, 0.02),
        n_T=args.diffusion_steps,
        device=args.device,
        x_dim=x_shape,
        y_dim=y_dim,
        drop_prob=args.drop_prob,
        guide_w=args.cfg_weight,
    ).to(args.device)
    model.eval()

    dqn_agent = DDQN(state_dim = x_shape, z_dim=y_dim, diffusion_prior=model)
    dqn_agent.learn(dataload_train=dataload_train, diffusion_model_name=args.skill_model_filename[:-4])


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--net_type', type=str, default='unet')
    parser.add_argument('--n_hidden', type=int, default=512)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--sample_z', type=int, default=0)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--skill_model_filename', type=str)

    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--predict_noise', type=int, default=0)

    args = parser.parse_args()

    train(args)
