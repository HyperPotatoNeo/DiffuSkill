from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from models.skill_model import SkillModel
import random
import gym
import d4rl
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
from math import pi
from planning.cem import cem
import itertools
from statsmodels.stats.proportion import proportion_confint

p_succ = 0 #Incase need to resume experiments
p_n_tot = 0 #Incase need to resume experiments

device = torch.device('cuda:0')

env = 'antmaze-large-diverse-v0'
#env = 'antmaze-medium-diverse-v0'
#env = 'maze2d-large-v1'

env_name = env
env = gym.make(env)
data = env.get_dataset()

goal_conditioned = False#True

skill_seq_len = 5
H = 40
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
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

max_replans = 2000
n_iters = 10
batch_size = 100
keep_frac = 0.5
cem_l2_pen = 0.0
random_goal = False#True # determines if we select a goal at random from dataset (random_goal=True) or use pre-set one from environment
tanh_normal = True
distribution_loss = True

filename = 'EM_model_antmaze-large-diverse-v0state_dec_mlp_H_40_l2reg_0.0_b_1.0_per_el_sig_False_log_best.pth'

PATH = 'checkpoints/'+filename

skill_model = SkillModel(state_dim,a_dim,z_dim,h_dim,a_dist=a_dist,beta=beta,fixed_sig=None,encoder_type=encoder_type,state_decoder_type=state_decoder_type,policy_decoder_type=policy_decoder_type,per_element_sigma=per_element_sigma).cuda()

checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'DiffuSkill')

s0_torch = torch.cat([torch.tensor(env.reset(),dtype=torch.float32).cuda().reshape(1,1,-1) for _ in range(batch_size)])

skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
print('skill_seq.shape: ', skill_seq.shape)
skill_seq.requires_grad = True


def convert_epsilon_to_z(epsilon,s0,model):
	s = s0
	z_seq = []
	for i in range(epsilon.shape[1]):
		# get prior
		mu_z, sigma_z = model.prior(s)
		z_i = mu_z + sigma_z*skill_seq[:,i:i+1,:]
		z_seq.append(z_i)
		s_mean,_ = model.decoder.abstract_dynamics(s,z_i)
		s = s_mean

	return torch.cat(z_seq,dim=1)

def run_skill_seq(skill_seq,env,s0,model,goal=None):
	state = s0

	pred_states = []
	pred_sigs = []
	states = []
	# plt.figure()
	for i in range(execute_n_skills):
		# get the skill
		mu_z,sig_z = model.prior(torch.tensor(state,dtype=torch.float32).cuda().reshape(1,1,-1),goal)
		z = mu_z + sig_z*skill_seq[:,i:i+1,:]
		skill_seq_states = []
		
		# run skill for H timesteps
		for j in range(H):
			env.render()
			#state_goal = np.hstack([state,goal_state])
			#action = model.decoder.ll_policy.numpy_policy(state_goal,z)
			action = model.decoder.ll_policy.numpy_policy(state,z)
			state,_,_,_ = env.step(action)
			skill_seq_states.append(state)
		states.append(skill_seq_states)

	states = np.stack(states)

	return state,states

execute_n_skills = 1

min_dists_list = []
for j in range(1000):
	env.set_target() # this randomizes goal locations between trials, so that we're actualy averaging over the goal distribution
	# otherwise, same goal is kept across resets
	if not random_goal:
		
		goal_state = np.array(env.target_goal)#random.choice(data['observations'])
		print('goal_state: ', goal_state)
	else:
		N = data['observations'].shape[0]
		ind = np.random.randint(low=0,high=N)
		goal_state = data['observations'][ind,:]
		print('goal_state: ', goal_state[:2])
	goal_seq = torch.tensor(goal_state, device=device).reshape(1,1,-1)

	state = env.reset()
	goal_loc = goal_state[:2]
	min_dist = 10**10
	for i in range(max_replans):
		if(i%50==0):
			print(i,'/',max_replans)
		s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))]).cuda()
		
		cost_fn = lambda skill_seq: skill_model.get_expected_cost_antmaze(s_torch, skill_seq, goal_seq)
		skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)

		skill_seq = skill_seq[:execute_n_skills,:]
		skill_seq = skill_seq.unsqueeze(0)
		#skill_seq = None
		state,states = run_skill_seq(skill_seq,env,state,skill_model,goal_seq.float())

		dists = np.sqrt(np.sum((states[0,:,:2] - goal_loc)**2,axis=-1))

		if np.min(dists) < min_dist:
			min_dist = np.min(dists)

		if min_dist <= 0.5:
			break
		if(i%10==0):
			print(min_dist)
	min_dists_list.append(min_dist)

	p_succ = 0 #Incase need to resume experiments
	p_n_tot = 0 #Incase need to resume experiments
	n_success = np.sum(np.array(min_dists_list) <= 0.5)
	n_tot = len(min_dists_list)

	ci = proportion_confint(n_success+p_succ,n_tot+p_n_tot)
	print('ci: ', ci)
	print('mean: ',(n_success+p_succ)/(n_tot+p_n_tot))
	print('N = ',n_tot+p_n_tot)
	print('n_success = ,',n_success+p_succ)