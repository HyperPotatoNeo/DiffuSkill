import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

class MLP_policy(nn.Module):
    def __init__(self,state_dim,z_dim,h_dim=256):
        super(MLP_policy,self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, z_dim)
        )


    def forward(self,s):
        '''
        INPUTS:
            s: batch_size x state_dim
        OUTPUS:
            z: batch_size x z_dim
        '''
        z = self.mlp(s)
        return z

class Autoregressive_policy(nn.Module):
    def __init__(self,state_dim,z_dim,h_dim=256):

        super(Autoregressive_policy,self).__init__()
        self.policy_components = nn.ModuleList([MLP_policy(state_dim+i,1,h_dim) for i in range(a_dim)])
        self.state_dim = state_dim
        
    def forward(self,state):
        latents = []
        for i in range(self.state_dim):
            state_z = torch.cat([state]+latents,dim=-1)
            z = self.policy_components[i](state_z)
            latents.append(z)
        return torch.cat(latents)