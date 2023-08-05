import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
from models.q_learning_models import MLP_Q
from comet_ml import Experiment
import copy
from tqdm import tqdm


class ImageStateEncoder(nn.Module):
    def __init__(self, state_dim=64, horizon=10):
        super(ImageStateEncoder, self).__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.conv1 = nn.Conv2d(4, 8, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gelu = nn.GELU()
        self.ln1 = nn.LayerNorm(normalized_shape=(8, 80, 80))
        self.ln2 = nn.LayerNorm(normalized_shape=(16, 19, 19))
        self.ln3 = nn.LayerNorm(normalized_shape=(32, 7, 7))
        self.ln4 = nn.LayerNorm(normalized_shape=(64, 3, 3))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size*self.horizon, 4, 84, 84)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.ln3(x)
        x = self.gelu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.ln4(x)
        x = self.maxpool(x)
        x = x.reshape(batch_size, self.horizon, self.state_dim)
        return x


class ImageStateDecoder(nn.Module):
    def __init__(self, state_dim=64, horizon=10):
        super(ImageStateDecoder, self).__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.upsample1 = nn.Upsample(size=(4,4))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.upsample2 = nn.Upsample(size=(8,8))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.upsample3 = nn.Upsample(size=(16,16))
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.upsample4 = nn.Upsample(size=(32,32))
        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, stride=1)
        self.upsample5 = nn.Upsample(size=(64,64))
        self.conv6 = nn.Conv2d(8, 8, kernel_size=3, stride=1)
        self.upsample6 = nn.Upsample(size=(88,88))
        self.conv7 = nn.Conv2d(8, 4, kernel_size=3, stride=1)
        self.gelu = nn.GELU()
        self.ln1 = nn.LayerNorm(normalized_shape=(64, 2, 2))
        self.ln2 = nn.LayerNorm(normalized_shape=(32, 6, 6))
        self.ln3 = nn.LayerNorm(normalized_shape=(32, 14, 14))
        self.ln4 = nn.LayerNorm(normalized_shape=(16, 30, 30))
        self.ln5 = nn.LayerNorm(normalized_shape=(8, 62, 62))
        self.ln6 = nn.LayerNorm(normalized_shape=(8, 86, 86))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size*self.horizon, self.state_dim, 1, 1)
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.ln3(x)
        x = self.gelu(x)
        x = self.upsample4(x)
        x = self.conv4(x)
        x = self.ln4(x)
        x = self.gelu(x)
        x = self.upsample5(x)
        x = self.conv5(x)
        x = self.ln5(x)
        x = self.gelu(x)
        x = self.upsample6(x)
        x = self.conv6(x)
        x = self.ln6(x)
        x = self.gelu(x)
        x = self.conv7(x)
        x = x.reshape(batch_size, self.horizon, 4, 84, 84)
        return x


class InverseDynamics(nn.Module):
    def __init__(self, state_dim=64, a_dim=2):
        super(InverseDynamics, self).__init__()
        self.state_dim = 64
        self.a_dim = a_dim
        self.inv_dynamics_net = nn.Sequential(nn.Linear(2*state_dim,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,a_dim),nn.Softmax(dim=2))
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.inv_dynamics_net(x)
        return x

    def get_losses(self, states, actions):
        states_cat = torch.cat([states[:,:-1,:],states[:,1:,:]],dim=2)
        actions = actions[:,:-1,:]
        actions_pred = self.forward(states_cat)
        loss = self.CELoss(actions_pred, actions)/(states.shape[0]*states.shape[1])
        return loss


class ImageRepresentation(nn.Module):
    def __init__(self, state_dim=64, a_dim=2, horizon=10, noise_std=0.05):
        super(ImageRepresentation, self).__init__()
        self.state_dim = state_dim
        self.a_dim = a_dim
        self.horizon = horizon
        self.encoder = ImageStateEncoder(state_dim, horizon)
        self.decoder = ImageStateDecoder(state_dim, horizon)
        self.inv_dynamics_model = InverseDynamics(state_dim, a_dim)
        self.noise_std = noise_std

    def forward(self, states):
        z = self.encoder(states)
        states_cat = torch.cat([z[:,:-1,:],z[:,1:,:]],dim=2)
        a_pred = self.inv_dynamics_model(states_cat)
        states_pred = self.decoder(z)
        return z, a_pred, states_pred

    def get_losses(self, states, actions):
        actions = torch.nn.functional.one_hot(torch.squeeze(actions,dim=2), num_classes=self.a_dim)[:,:-1,:].float()
        noised_states = states + self.noise_std*torch.randn(size=states.shape).cuda()
        z, a_pred, states_pred = self.forward(noised_states)
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        a_loss = ce_loss(a_pred, actions)/(actions.shape[0]*actions.shape[1])
        s_loss = mse_loss(states_pred, states)
        total_loss = a_loss + s_loss
        return total_loss, a_loss, s_loss
