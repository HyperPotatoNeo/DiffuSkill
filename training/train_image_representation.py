from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from models.image_representation import ImageRepresentation
import h5py
from utils.utils import get_dataset, StateActionReturnDataset
import os
import pickle
import argparse


def train(model, optimizer):
    losses = []
    a_losses = []
    s_losses = []
    
    for batch_id, data in enumerate(train_loader):
        states = torch.reshape(data[0].cuda(),(-1,H,4,84,84))
        actions = data[1].cuda()
        loss_tot, a_loss, s_loss = model.get_losses(states, actions)
        model.zero_grad()
        loss_tot.backward()
        optimizer.step()
        # log losses
        losses.append(loss_tot.item())
        a_losses.append(a_loss.item())
        s_losses.append(s_loss.item())
        
    return np.mean(losses), np.mean(a_losses), np.mean(s_losses)


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='atari-Breakout')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--horizon', type=int, default=30)
args = parser.parse_args()

env_name = args.env_name
H = args.horizon
lr = args.lr
batch_size = args.batch_size
checkpoint_dir = 'checkpoints/'

obss, actions, rewards, done_idxs, timesteps = get_dataset(env_name, H, 1, 0.0, get_rewards=1, separate_test_trajectories=0, append_goals=0)
state_dim = 64
a_dim = max(actions) + 1

filename = 'image_encoder_'+env_name+'_H_'+str(H)
experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'DiffuSkill')

model = ImageRepresentation(state_dim, a_dim, H).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

experiment.log_parameters({'lr':lr,
                            'batch_size':batch_size,
                            'H':H,
                            'env_name':env_name})
inputs_train = StateActionReturnDataset(obss, H*3, actions, done_idxs, rewards, timesteps)
train_loader = DataLoader(inputs_train, shuffle=True, pin_memory=True,
                            batch_size=batch_size,
                            num_workers=4)

for i in range(1000000):
    print(i)
    loss, train_a_loss, train_s_loss = train(model, optimizer)
    print("--------TRAIN---------")
    print('Loss: ', loss)
    print('train_a_loss: ', train_a_loss)
    print('train_s_loss: ', train_s_loss)
    experiment.log_metric("Train loss", loss, step=i)
    experiment.log_metric("test_a_loss", train_a_loss, step=i)
    experiment.log_metric("test_s_loss", train_s_loss, step=i)

    if i % 50 == 0:
        checkpoint_path = os.path.join(checkpoint_dir,filename+'_'+str(i)+'_'+'.pth')
        torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
