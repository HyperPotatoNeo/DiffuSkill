from argparse import ArgumentParser
import os
from comet_ml import Experiment

import gym
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from models.dqn import DDQN

class QLearningDataset(Dataset):
    def __init__(
        self, dataset_dir, filename, train_or_test="train", test_prop=0.1, sample_z=False
    ):
        # just load it all into RAM
        self.state_all = np.load(os.path.join(dataset_dir, filename + "_states.npy"), allow_pickle=True)
        self.latent_all = np.load(os.path.join(dataset_dir, filename + "_latents.npy"), allow_pickle=True)
        self.sT_all = np.load(os.path.join(dataset_dir, filename + "_sT.npy"), allow_pickle=True)
        self.rewards_all = np.load(os.path.join(dataset_dir, filename + "_rewards.npy"), allow_pickle=True)#(4*np.load(os.path.join(dataset_dir, filename + "_rewards.npy"), allow_pickle=True) - 30*4*0.5)/10 #zero-centering
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

@torch.no_grad()
def collect_advantage_data(dqn_agent, filename, q_checkpoint_steps):
    print('collecting data:')
    dataset_dir = 'data/'
    state_all = np.load(os.path.join(dataset_dir, filename + "_states.npy"), allow_pickle=True)
    latent_all = np.load(os.path.join(dataset_dir, filename + "_latents.npy"), allow_pickle=True)
    sample_latents = np.load(os.path.join(dataset_dir, filename + "_sample_latents.npy"), allow_pickle=True)

    advantage_data = np.zeros((state_all.shape[0],1))
    batch_size = 64
    for i in range(int(np.ceil(state_all.shape[0]/batch_size))):
        #print(i,int(np.ceil(state_all.shape[0]/batch_size)))
        idx_start = i*batch_size
        idx_end = min(i*batch_size + batch_size, state_all.shape[0])
        states = torch.tensor(state_all[idx_start:idx_end]).float().cuda()
        latents = torch.tensor(latent_all[idx_start:idx_end]).float().cuda()
        sample_z = torch.tensor(sample_latents[idx_start:idx_end]).float().cuda()
        sample_z = sample_z.reshape(sample_z.shape[0]*sample_z.shape[1],sample_z.shape[2])
        states_interleaved = states.repeat_interleave(sample_latents.shape[1], 0)

        q_val_0 = dqn_agent.q_net_0(states, latents)[:,0]
        q_val_1 = dqn_agent.q_net_1(states, latents)[:,0]
        q_val = torch.minimum(q_val_0, q_val_1)
        v_val_0 = dqn_agent.q_net_0(states_interleaved, sample_z)[:,0].reshape(states.shape[0], sample_latents.shape[1])
        v_val_1 = dqn_agent.q_net_1(states_interleaved, sample_z)[:,0].reshape(states.shape[0], sample_latents.shape[1])
        v_val = torch.minimum(v_val_0, v_val_1)
        v_val = torch.mean(v_val, dim=1)
        advantage = (q_val-v_val).cpu().numpy()
        advantage_data[idx_start:idx_end,0] = advantage
    np.save('data/'+filename+'_dqn_agent_'+str(args.q_checkpoint_steps)+'_advantage.npy', advantage_data)

def train(args):
    dataset_file = 'data/'+args.env+'.pkl'
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)
    state_dim = dataset['observations'].shape[1]
    a_dim = dataset['actions'].shape[1]

    dqn_agent = torch.load(os.path.join(args.q_checkpoint_dir, args.skill_model_filename[:-4]+'_dqn_agent_'+str(args.q_checkpoint_steps)+'_cfg_weight_'+str(args.cfg_weight)+'_PERbuffer.pt')).to(args.device)
    dqn_agent.extra_steps = 5#args.extra_steps
    dqn_agent.eval()

    if args.collect_data:
        collect_advantage_data(dqn_agent, args.skill_model_filename[:-4], args.q_checkpoint_steps)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--test_split', type=float, default=0.0)
    parser.add_argument('--total_prior_samples', type=int, default=1000)
    parser.add_argument('--collect_data', type=int, default=1)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--q_checkpoint_dir', type=str, default='q_checkpoints')
    parser.add_argument('--awr_checkpoint_dir', type=str, default='awr_checkpoints')
    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--q_checkpoint_steps', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1.0)

    args = parser.parse_args()

    train(args)
