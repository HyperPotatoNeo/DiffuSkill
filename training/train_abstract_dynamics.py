from argparse import ArgumentParser
import os
from comet_ml import Experiment

import d4rl
import gym
import pickle
import numpy as np

import torch
import torch.distributions.normal as Normal
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.skill_model import SkillModel, AbstractDynamics, AutoregressiveStateDecoder


class StateDecoderDataset(Dataset):
    def __init__(
        self, dataset_dir, filename, train_or_test="train", train_prop=0.90, sample_z=False
    ):
        # just load it all into RAM
        self.state_all = np.load(os.path.join(dataset_dir, filename + "_states.npy"), allow_pickle=True)
        self.latent_all = np.load(os.path.join(dataset_dir, filename + "_latents.npy"), allow_pickle=True)
        self.sT_all = np.load(os.path.join(dataset_dir, filename + "_sT.npy"), allow_pickle=True)
        self.sample_z = sample_z
        if sample_z:
            self.latent_all_std = np.load(os.path.join(dataset_dir, filename + "_latents_std.npy"), allow_pickle=True)
        n_train = int(self.state_all.shape[0] * train_prop)
        if train_or_test == "train":
            self.state_all = self.state_all[:n_train]
            self.latent_all = self.latent_all[:n_train]
            self.sT_all = self.sT_all[:n_train]
        elif train_or_test == "test":
            self.state_all = self.state_all[n_train:]
            self.latent_all = self.latent_all[n_train:]
            self.sT_all = self.sT_all[n_train:]
        else:
            raise NotImplementedError

        self.state_mean = self.state_all.mean(axis=0)
        self.state_std = self.state_all.std(axis=0)
        #self.state_all = (self.state_all - self.state_mean) / self.state_std
        #Normalize sT with s0 statistics
        #self.sT_all = (self.sT_all - self.state_mean) / self.state_std

        self.latent_mean = self.latent_all.mean(axis=0)
        self.latent_std = self.latent_all.std(axis=0)
        #self.latent_all = (self.latent_all - self.latent_mean) / self.latent_std

    def __len__(self):
        return self.state_all.shape[0]

    def __getitem__(self, index):
        state = self.state_all[index]
        latent = self.latent_all[index]
        if self.sample_z:
            latent_std = self.latent_all_std[index]
            latent = np.random.normal(latent,latent_std)
            #latent = (latent - self.latent_mean) / self.latent_std
        sT = self.sT_all[index]

        return (state, latent, sT)


def train(args):
    env = gym.make(args.env)
    dataset = env.get_dataset()
    state_dim = dataset['observations'].shape[1]
    a_dim = dataset['actions'].shape[1]

    experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'DiffuSkill')
    experiment.log_parameters({'lrate':args.lrate,
                            'batch_size':args.batch_size,
                            'net_type':args.net_type,
                            'sample_z':args.sample_z,
                            'state_decoder_type':args.state_decoder_type,
                            'skill_model_filename':args.skill_model_filename,
                            'z_dim':args.z_dim})

    # get datasets set up
    torch_data_train = StateDecoderDataset(
        args.dataset_dir, args.skill_model_filename[:-4], train_or_test="train", train_prop=0.90, sample_z=args.sample_z
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    torch_data_test = StateDecoderDataset(
        args.dataset_dir, args.skill_model_filename[:-4], train_or_test="test", train_prop=0.90, sample_z=args.sample_z
    )
    dataload_test = DataLoader(
        torch_data_test, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # load model
    skill_model_path = os.path.join(args.checkpoint_dir, args.skill_model_filename)
    checkpoint = torch.load(skill_model_path)
    skill_model = SkillModel(state_dim,
                             a_dim,
                             args.z_dim,
                             args.h_dim,
                             args.horizon,
                             a_dist=args.a_dist,
                             beta=args.beta,
                             fixed_sig=None,
                             encoder_type=args.encoder_type,
                             state_decoder_type=args.state_decoder_type,
                             policy_decoder_type=args.policy_decoder_type,
                             per_element_sigma=args.per_element_sigma,
                             conditional_prior=args.conditional_prior,
                             ).to(args.device)
    skill_model.load_state_dict(checkpoint['model_state_dict'])

    if args.state_decoder_type == 'mlp':
        skill_model.decoder.abstract_dynamics = AbstractDynamics(state_dim,args.z_dim,args.h_dim,per_element_sigma=args.per_element_sigma).to(args.device)
    elif args.state_decoder_type == 'autoregressive':
        skill_model.decoder.abstract_dynamics = AutoregressiveStateDecoder(state_dim,args.z_dim,args.h_dim).to(args.device)

    optim = torch.optim.Adam(skill_model.decoder.abstract_dynamics.parameters(), lr=args.lrate)
    best_test_loss = 10000000

    for ep in tqdm(range(args.n_epoch), desc="Epoch"):
        results_ep = [ep]
        skill_model.decoder.abstract_dynamics.train()

        # train loop
        pbar = tqdm(dataload_train)
        loss_ep, n_batch = 0, 0
        
        for state, latent, sT in pbar:
            state = state.type(torch.FloatTensor).to(args.device)
            latent = latent.type(torch.FloatTensor).to(args.device)
            sT = sT.type(torch.FloatTensor).to(args.device)
            if args.state_decoder_type == 'mlp':
                sT_pred_mean, sT_pred_sig = skill_model.decoder.abstract_dynamics(state.unsqueeze(1),latent.unsqueeze(1))
            elif args.state_decoder_type == 'autoregressive':
                sT_pred_mean, sT_pred_sig = skill_model.decoder.abstract_dynamics(state.unsqueeze(1),sT.unsqueeze(1),latent.unsqueeze(1))
            s_T_dist = Normal.Normal(sT_pred_mean.squeeze(1), sT_pred_sig.squeeze(1))
            loss = -torch.mean(torch.sum(s_T_dist.log_prob(sT), dim=-1))
            optim.zero_grad()
            loss.backward()
            loss_ep += loss.detach().item()
            n_batch += 1
            pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
            optim.step()

        experiment.log_metric("train_loss", loss_ep/n_batch, step=ep)
        results_ep.append(loss_ep / n_batch)
        
        # test loop
        skill_model.decoder.abstract_dynamics.eval()
        pbar = tqdm(dataload_test)
        loss_ep, n_batch = 0, 0

        for state, latent, sT in pbar:
            state = state.type(torch.FloatTensor).to(args.device)
            latent = latent.type(torch.FloatTensor).to(args.device)
            sT = sT.type(torch.FloatTensor).to(args.device)
            if args.state_decoder_type == 'mlp':
                sT_pred_mean, sT_pred_sig = skill_model.decoder.abstract_dynamics(state.unsqueeze(1),latent.unsqueeze(1))
            elif args.state_decoder_type == 'autoregressive':
                sT_pred_mean, sT_pred_sig = skill_model.decoder.abstract_dynamics(state.unsqueeze(1),None,latent.unsqueeze(1),evaluation=True)
            s_T_dist = Normal.Normal(sT_pred_mean.squeeze(1), sT_pred_sig.squeeze(1))
            loss = -torch.mean(torch.sum(s_T_dist.log_prob(sT), dim=-1))
            loss_ep += loss.detach().item()
            n_batch += 1
            pbar.set_description(f"test loss: {loss_ep/n_batch:.4f}")

        experiment.log_metric("test_loss", loss_ep/n_batch, step=ep)

        if loss_ep < best_test_loss:
            best_test_loss = loss_ep
            torch.save({'model_state_dict': skill_model.state_dict()}, os.path.join(args.checkpoint_dir, args.skill_model_filename))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sample_z', type=int, default=0)

    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='autoregressive')
    parser.add_argument('--policy_decoder_type', type=str, default='autoregressive')
    parser.add_argument('--per_element_sigma', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=64)

    args = parser.parse_args()

    train(args)
