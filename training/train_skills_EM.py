from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from models.skill_model import SkillModel
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import d4rl
import ipdb
import h5py
from utils.utils import chunks
import os
import pickle

def train(model,E_optimizer,M_optimizer):
	
	E_losses = []
	M_losses = []
	
	for batch_id, data in enumerate(train_loader):
		data = data.cuda()
		states = data[:,:,:model.state_dim]
		actions = data[:,:,model.state_dim:]	 # rest are actions

		########### E STEP ###########
		E_loss = model.get_E_loss(states,actions)
		model.zero_grad()
		E_loss.backward()
		E_optimizer.step()

		########### M STEP ###########
		M_loss = model.get_M_loss(states,actions)
		model.zero_grad()
		M_loss.backward()
		M_optimizer.step()

		# log losses
		E_losses.append(E_loss.item())
		M_losses.append(M_loss.item())
		
	return np.mean(E_losses),np.mean(M_losses)

def test(model):
	
	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	s_T_ents = []

	with torch.no_grad():
		for batch_id, data in enumerate(test_loader):
			data = data.cuda()
			states = data[:,:,:model.state_dim]  # first state_dim elements are the state
			actions = data[:,:,model.state_dim:]	 # rest are actions

			loss_tot, s_T_loss, a_loss, kl_loss  = model.get_losses(states, actions)

			# log losses
			losses.append(loss_tot.item())
			s_T_losses.append(s_T_loss.item())
			a_losses.append(a_loss.item())
			kl_losses.append(kl_loss.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses)

batch_size = 100

h_dim = 256
z_dim = 256
lr = 5e-5
wd = 0.0
beta = 1.0
H = 40
stride = 1
n_epochs = 50000
test_split = .2
a_dist = 'normal' # 'tanh_normal' or 'normal'
encoder_type = 'gru' #'state_sequence'
state_decoder_type = 'mlp' #'autoregressive'
policy_decoder_type = 'autoregressive'
load_from_checkpoint = False
per_element_sigma = False

env_name = 'antmaze-large-diverse-v0'
#env_name = 'kitchen-partial-v0'

dataset_file = 'data/'+env_name+'.pkl'
with open(dataset_file, "rb") as f:
	dataset = pickle.load(f)

checkpoint_dir = 'checkpoints/'
states = dataset['observations'][:10000]
#next_states = dataset['next_observations']
actions = dataset['actions'][:10000]

N = states.shape[0]

state_dim = states.shape[1]
a_dim = actions.shape[1]

N_train = int((1-test_split)*N)
N_test = N - N_train

states_train  = states[:N_train,:]
#next_states_train = next_states[:N_train,:]
actions_train = actions[:N_train,:]

states_test  = states[N_train:,:]
#next_states_test = next_states[N_train:,:]
actions_test = actions[N_train:,:]

obs_chunks_train, action_chunks_train = chunks(states_train, actions_train, H, stride)

print('states_test.shape: ',states_test.shape)
print('MAKIN TEST SET!!!')

obs_chunks_test,  action_chunks_test  = chunks(states_test, actions_test, H, stride)

experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'DiffuSkill')
# experiment.add_tag('noisy2')

model = SkillModel(state_dim,a_dim,z_dim,h_dim,a_dist=a_dist,beta=beta,fixed_sig=None,encoder_type=encoder_type,state_decoder_type=state_decoder_type,policy_decoder_type=policy_decoder_type,per_element_sigma=per_element_sigma).cuda()
	
E_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=lr, weight_decay=wd)
M_optimizer = torch.optim.Adam(model.gen_model.parameters(), lr=lr, weight_decay=wd)

filename = 'EM_model_'+env_name+'state_dec_'+str(state_decoder_type)+'_H_'+str(H)+'_l2reg_'+str(wd)+'_b_'+str(beta)+'_per_el_sig_'+str(per_element_sigma)+'_log'

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
							'per_element_sigma':per_element_sigma})

inputs_train = torch.cat([obs_chunks_train, action_chunks_train],dim=-1)
inputs_test  = torch.cat([obs_chunks_test,  action_chunks_test], dim=-1)


train_data = TensorDataset(inputs_train)
test_data  = TensorDataset(inputs_test)

train_loader = DataLoader(
	inputs_train,
	batch_size=batch_size,
	num_workers=0)  # not really sure about num_workers...

test_loader = DataLoader(
	inputs_test,
	batch_size=batch_size,
	num_workers=0)

min_test_loss = 10**10
min_test_s_T_loss = 10**10
min_test_a_loss = 10**10
for i in range(n_epochs):

	test_loss, test_s_T_loss, test_a_loss, test_kl_loss = test(model)
	
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
				'E_optimizer_state_dict': E_optimizer.state_dict(),
							'M_optimizer_state_dict': M_optimizer.state_dict()}, checkpoint_path)

	if test_s_T_loss < min_test_s_T_loss:
		min_test_s_T_loss = test_s_T_loss

		checkpoint_path = os.path.join(checkpoint_dir,filename+'_best_sT.pth')
		torch.save({'model_state_dict': model.state_dict(),
				'E_optimizer_state_dict': E_optimizer.state_dict(),
							'M_optimizer_state_dict': M_optimizer.state_dict()}, checkpoint_path)

	if test_a_loss < min_test_a_loss:
		min_test_a_loss = test_a_loss

		checkpoint_path = os.path.join(checkpoint_dir,filename+'_best_a.pth')
		torch.save({'model_state_dict': model.state_dict(),
				'E_optimizer_state_dict': E_optimizer.state_dict(),
							'M_optimizer_state_dict': M_optimizer.state_dict()}, checkpoint_path)

	E_loss,M_loss = train(model,E_optimizer,M_optimizer)
	
	print("--------TRAIN---------")
	
	print('E_loss: ', E_loss)
	print('M_loss: ', M_loss)
	print(i)
	experiment.log_metric("E_loss", E_loss, step=i)
	experiment.log_metric("M_loss", M_loss, step=i)

	if i % 10 == 0:
		
		checkpoint_path = os.path.join(checkpoint_dir,filename+'.pth')
		torch.save({
							'model_state_dict': model.state_dict(),
							'E_optimizer_state_dict': E_optimizer.state_dict(),
							'M_optimizer_state_dict': M_optimizer.state_dict()
							}, checkpoint_path)

	