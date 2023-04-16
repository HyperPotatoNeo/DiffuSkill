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

class DDQN(nn.Module):
    def __init__(self, state_dim, z_dim, h_dim=256, gamma=0.995, tau=0.995, lr=1e-3, num_prior_samples=100, extra_steps=10, device='cuda', diffusion_prior=None):
        super(DDQN,self).__init__()

        self.state_dim = state_dim
        self.z_dim = z_dim
        self.gamma = gamma
        self.lr = lr
        self.num_prior_samples = num_prior_samples
        self.extra_steps = extra_steps
        self.device = device
        self.tau = tau
        self.diffusion_prior = diffusion_prior

        self.q_net_0 = MLP_Q(state_dim=state_dim,z_dim=z_dim,h_dim=h_dim).to(self.device)
        self.q_net_1 = MLP_Q(state_dim=state_dim,z_dim=z_dim,h_dim=h_dim).to(self.device)
        self.target_net_0 = None
        self.target_net_1 = None
        
        self.optimizer_0 = optim.Adam(params=self.q_net_0.parameters(), lr=lr)
        self.optimizer_1 = optim.Adam(params=self.q_net_1.parameters(), lr=lr)
        self.scheduler_0 = optim.lr_scheduler.StepLR(self.optimizer_0, step_size=2, gamma=0.3)
        self.scheduler_1 = optim.lr_scheduler.StepLR(self.optimizer_1, step_size=2, gamma=0.3)

    @torch.no_grad()
    def get_max_skills(self, states, net=0):
        '''
        INPUTS:
            states: batch_size x state_dim
        OUTPUTS:
            max_z: batch_size x z_dim
        '''
        n_states = states.shape[0]
        states = states.repeat_interleave(self.num_prior_samples, 0)
        z_samples = self.diffusion_prior.sample_extra(states, predict_noise=0, extra_steps=self.extra_steps)
        if net==0:
            q_vals = self.target_net_0(states,z_samples)[:,0]#self.q_net_0(states,z_samples)[:,0]
        else:
            q_vals = self.target_net_1(states,z_samples)[:,0]#self.q_net_1(states,z_samples)[:,0]
        q_vals = q_vals.reshape(n_states, self.num_prior_samples)
        max_vals = torch.max(q_vals, dim=1)
        max_q_vals = max_vals.values
        max_indices = max_vals.indices
        idx = torch.arange(n_states).cuda()*self.num_prior_samples + max_indices 
        max_z = z_samples[idx]

        return max_z, max_q_vals


    def learn(self, dataload_train, dataload_test=None, n_epochs=10000, update_frequency=50, diffusion_model_name='', cfg_weight=0.0):
        assert self.diffusion_prior is not None
        experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'DiffuSkill')
        experiment.log_parameters({'diffusion_prior':diffusion_model_name, 'cfg_weight':cfg_weight})
        steps_net_0, steps_net_1, steps_total = 0, 0, 0
        self.target_net_0 = copy.deepcopy(self.q_net_0)
        self.target_net_1 = copy.deepcopy(self.q_net_1)
        self.target_net_0.eval()
        self.target_net_1.eval()
        loss_net_0, loss_net_1, loss_total = 0, 0, 0 #Logged in comet at update frequency
        epoch = 0

        for ep in tqdm(range(n_epochs), desc="Epoch"):
            n_batch = 0
            loss_ep = 0
            self.q_net_0.train()
            self.q_net_1.train()
            pbar = tqdm(dataload_train)

            for s0,z,sT,reward in pbar:
                s0 = s0.type(torch.FloatTensor).to(self.device)
                z = z.type(torch.FloatTensor).to(self.device)
                sT = sT.type(torch.FloatTensor).to(self.device)
                reward = reward.type(torch.FloatTensor).to(self.device)
                net_id = np.random.binomial(n=1, p=0.5, size=(1,))

                if net_id==0:
                    self.optimizer_0.zero_grad()

                    q_s0z = self.q_net_0(s0,z)
                    max_sT_skills,_ = self.get_max_skills(sT,net=1-net_id)
                    q_sTz = torch.minimum(self.target_net_0(sT,max_sT_skills.detach()),
                                          self.target_net_1(sT,max_sT_skills.detach()),)
                    q_target = reward + self.gamma*q_sTz
                    
                    bellman_loss = F.mse_loss(q_s0z, q_target)
                    bellman_loss.backward()
                    clip_grad_norm_(self.q_net_0.parameters(), 1)
                    self.optimizer_0.step()
                    loss_net_0 += bellman_loss.detach().item()
                    loss_total += bellman_loss.detach().item()
                    loss_ep += bellman_loss.detach().item()
                    steps_net_0 += 1
                else:
                    self.optimizer_1.zero_grad()

                    q_s0z = self.q_net_1(s0,z)
                    max_sT_skills,_ = self.get_max_skills(sT,net=1-net_id)
                    q_sTz = torch.minimum(self.target_net_0(sT,max_sT_skills.detach()),
                                          self.target_net_1(sT,max_sT_skills.detach()),)
                    q_target = reward + self.gamma*q_sTz
                    
                    bellman_loss = F.mse_loss(q_s0z, q_target)
                    bellman_loss.backward()
                    clip_grad_norm_(self.q_net_1.parameters(), 1)
                    self.optimizer_1.step()
                    loss_net_1 += bellman_loss.detach().item()
                    loss_total += bellman_loss.detach().item()
                    loss_ep += bellman_loss.detach().item()
                    steps_net_1 += 1

                n_batch += 1
                steps_total += 1
                pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")

                if steps_total%update_frequency == 0:
                    loss_net_0 /= (steps_net_0+1e-4)
                    loss_net_1 /= (steps_net_1+1e-4)
                    loss_total /= update_frequency
                    experiment.log_metric("train_loss_0", loss_net_0, step=steps_total)
                    experiment.log_metric("train_loss_1", loss_net_1, step=steps_total)
                    experiment.log_metric("train_loss", loss_total, step=steps_total)
                    loss_net_0, loss_net_1, loss_total = 0,0,0
                    steps_net_0, steps_net_1 = 0,0
                    #self.target_net_0 = copy.deepcopy(self.q_net_0)
                    #self.target_net_1 = copy.deepcopy(self.q_net_1)
                    for target_param, local_param in zip(self.target_net_0.parameters(), self.q_net_0.parameters()):
                        target_param.data.copy_((1.0-self.tau)*local_param.data + (self.tau)*target_param.data)
                    for target_param, local_param in zip(self.target_net_1.parameters(), self.q_net_1.parameters()):
                        target_param.data.copy_((1.0-self.tau)*local_param.data + (self.tau)*target_param.data)
                    self.target_net_0.eval()
                    self.target_net_1.eval()
                    if steps_total%(5000) == 0:
                        torch.save(self,  'q_checkpoints_fixed/'+diffusion_model_name+'_dqn_agent_'+str(steps_total//5000)+'_cfg_weight_'+str(cfg_weight)+'.pt')

            self.scheduler_0.step()
            self.scheduler_1.step()
            experiment.log_metric("train_loss_episode", loss_ep/n_batch, step=epoch)
            epoch += 1
