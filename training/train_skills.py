from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from models.skill_model import SkillModel
import h5py
from utils.utils import get_dataset
import os
import pickle
import argparse

def train(model, optimizer, train_state_decoder):
	losses = []
	
	for batch_id, data in enumerate(train_loader):
		data = data.cuda()
		states = data[:,:,:model.state_dim]
		actions = data[:,:,model.state_dim:]
		if train_state_decoder:
			loss_tot, s_T_loss, a_loss, kl_loss = model.get_losses(states, actions, train_state_decoder)
		else:
			loss_tot, a_loss, kl_loss = model.get_losses(states, actions, train_state_decoder)
		model.zero_grad()
		loss_tot.backward()
		optimizer.step()
		# log losses
		losses.append(loss_tot.item())
		
	return np.mean(losses)

def test(model, test_state_decoder):
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	s_T_ents = []

	with torch.no_grad():
		for batch_id, data in enumerate(test_loader):
			data = data.cuda()
			states = data[:,:,:model.state_dim]
			actions = data[:,:,model.state_dim:]
			if test_state_decoder:
				loss_tot, s_T_loss, a_loss, kl_loss  = model.get_losses(states, actions, test_state_decoder)
				s_T_losses.append(s_T_loss.item())
			else:
				loss_tot, a_loss, kl_loss  = model.get_losses(states, actions, test_state_decoder)
			# log losses
			losses.append(loss_tot.item())
			a_losses.append(a_loss.item())
			kl_losses.append(kl_loss.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses)

batch_size = 100

h_dim = 256
z_dim = 64
lr = 5e-5
wd = 0.0
H = 30
stride = 1
n_epochs = 50000
test_split = .2
a_dist = 'normal' # 'tanh_normal' or 'normal'
encoder_type = 'gru' #'state_sequence'
state_decoder_type = 'autoregressive'
policy_decoder_type = 'autoregressive'
load_from_checkpoint = False
per_element_sigma = True
start_training_state_decoder_after = 0

parser = argparse.ArgumentParser()
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--conditional_prior', type=int, default=1)
args = parser.parse_args()

beta = args.beta # 1.0 # 0.1, 0.01, 0.001
conditional_prior = args.conditional_prior # True

env_name = 'antmaze-large-diverse-v2'
#env_name = 'kitchen-partial-v0'

dataset_file = 'data/'+env_name+'.pkl'
with open(dataset_file, "rb") as f:
	dataset = pickle.load(f)

checkpoint_dir = 'checkpoints/'
states = dataset['observations']
#next_states = dataset['next_observations']
actions = dataset['actions']

N = states.shape[0]

state_dim = states.shape[1]
a_dim = actions.shape[1]

N_train = int((1-test_split)*N)
N_test = N - N_train

dataset = get_dataset(env_name, H, stride, test_split)

obs_chunks_train = dataset['observations_train']
action_chunks_train = dataset['actions_train']
obs_chunks_test = dataset['observations_test']
action_chunks_test = dataset['actions_test']

experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'DiffuSkill')
#experiment.add_tag('noisy2')

model = SkillModel(state_dim,a_dim,z_dim,h_dim,a_dist=a_dist,beta=beta,fixed_sig=None,encoder_type=encoder_type,state_decoder_type=state_decoder_type,policy_decoder_type=policy_decoder_type,per_element_sigma=per_element_sigma, conditional_prior=conditional_prior).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

filename = 'skill_model_'+env_name+'_state_dec_'+str(state_decoder_type)+'_policy_dec_'+str(policy_decoder_type)+'_H_'+str(H)+'_b_'+str(beta)+'_conditionalp_'+str(conditional_prior)

if load_from_checkpoint:
	PATH = os.path.join(checkpoint_dir,filename+'_best_sT.pth')
	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint['model_state_dict'])
	E_optimizer.load_state_dict(checkpoint['E_optimizer_state_dict'])
	M_optimizer.load_state_dict(checkpoint['M_optimizer_state_dict'])

experiment.log_parameters({'lr':lr,
							'h_dim':h_dim,
							'z_dim':z_dim,
							'H':H,
							'a_dim':a_dim,
							'state_dim':state_dim,
							'l2_reg':wd,
							'beta':beta,
							'env_name':env_name,
							'filename':filename,
							'encoder_type':encoder_type,
							'state_decoder_type':state_decoder_type,
							'policy_decoder_type':policy_decoder_type,
							'per_element_sigma':per_element_sigma,
       						'conditional_prior': conditional_prior})

inputs_train = torch.cat([obs_chunks_train, action_chunks_train],dim=-1)
inputs_test  = torch.cat([obs_chunks_test,  action_chunks_test], dim=-1)

train_loader = DataLoader(
	inputs_train,
	batch_size=batch_size,
	num_workers=0)

test_loader = DataLoader(
	inputs_test,
	batch_size=batch_size,
	num_workers=0)

min_test_loss = 10**10
min_test_s_T_loss = 10**10
min_test_a_loss = 10**10
for i in range(n_epochs):

	test_loss, test_s_T_loss, test_a_loss, test_kl_loss = test(model, test_state_decoder = i > start_training_state_decoder_after)
	
	print("--------TEST---------")
	
	print('test_loss: ', test_loss)
	print('test_s_T_loss: ', test_s_T_loss)
	print('test_a_loss: ', test_a_loss)
	print('test_kl_loss: ', test_kl_loss)
	print(i)
	experiment.log_metric("test_loss", test_loss, step=i)
	experiment.log_metric("test_s_T_loss", test_s_T_loss, step=i)
	experiment.log_metric("test_a_loss", test_a_loss, step=i)
	experiment.log_metric("test_kl_loss", test_kl_loss, step=i)
	
	if test_loss < min_test_loss:
		min_test_loss = test_loss	
		checkpoint_path = os.path.join(checkpoint_dir,filename+'_best.pth')
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
	if test_s_T_loss < min_test_s_T_loss:
		min_test_s_T_loss = test_s_T_loss

		checkpoint_path = os.path.join(checkpoint_dir,filename+'_best_sT.pth')
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
	if test_a_loss < min_test_a_loss:
		min_test_a_loss = test_a_loss

		checkpoint_path = os.path.join(checkpoint_dir,filename+'_best_a.pth')
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

	loss = train(model, optimizer, train_state_decoder = i > start_training_state_decoder_after)
	
	print("--------TRAIN---------")
	
	print('Loss: ', loss)
	print(i)
	experiment.log_metric("Loss", loss, step=i)

	if i % 10 == 0:
		checkpoint_path = os.path.join(checkpoint_dir,filename+'.pth')
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)