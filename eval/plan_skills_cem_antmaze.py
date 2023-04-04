'''File where we will sample a set of waypionts, and plan a sequence of skills to have our pointmass travel through those waypoints'''

from tokenize import ContStr
from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior
import ipdb
import d4rl
import random
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from cem import cem, cem_variable_length
# from utils import make_video, save_frames_as_gif
# from gym.wrappers.monitoring import video_recorder
from utils import make_gif,make_video
from statsmodels.stats.proportion import proportion_confint

device = torch.device('cuda:0')

#env = 'antmaze-large-diverse-v0'
# env = 'antmaze-medium-diverse-v0'
env = 'maze2d-large-v1'
env_name = env
env = gym.make(env)
data = env.get_dataset()

# vid = video_recorder.VideoRecorder(env,path="recording")

skill_seq_len = 10
H = 10
replan_freq = H * 5
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 256
batch_size = 100
lr = 1e-4
wd = 0.0
state_dependent_prior = True
state_dec_stop_grad = True
beta = 1.0
alpha = 1.0
ent_pen = 0
max_sig = None
fixed_sig =  0.0
n_iters = 100
# n_iters = 200
a_dist = 'normal'
keep_frac = 0.5

# background_img = mpimg.imread('maze_medium.png')
use_epsilon = True
max_ep = None
cem_l2_pen = 0.0
var_pen = 0.0
render = False
variable_length = False
max_replans = 200
plan_length_cost = 0.0
encoder_type = 'state_action_sequence'
term_state_dependent_prior = False
init_state_dependent = True
# start_ind = 937278

# import glob
'''
if not state_dependent_prior:
  	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_sdp_'+str(state_dependent_prior)+'_log_best.pth'
else:
  	filename = 'AntMaze_H'+str(H)+'_l2reg_'+str(wd)+'_log_best.pth'
'''
# filename = 'maze2d_H'+str(H)+'_log_best.pth'
# filename = 'AntMaze_H20_l2reg_0.001_log_best.pth'
# filename = 'AntMaze_H20_l2reg_0.01_a_1.0_b_1.0_sg_True_max_sig_None_fixed_sig_0.1_log_best.pth'#'AntMaze_H20_l2reg_0.001_a_1.0_b_0.01_sg_False_log_best.pth'
# filename = 'AntMaze_H20_l2reg_0.001_log_best.pth'
# filename = 'AntMaze_H20_l2reg_0.0_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
# filename = 'AntMaze_H20_l2reg_0.0_a_1.0_b_0.1_sg_True_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
# filename = 'AntMaze_large_H20_l2reg_0.0_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
# filename = 'antmaze-large-diverse-v0_40_l2reg_0.0_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
# filename = 'antmaze-medium-diverse-v0_10_l2reg_0.0_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
# filename = 'antmaze-large-diverse-v0_20_l2reg_0.0_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
# filename = 'antmaze-large-diverse-v0_enc_type_state_sequence_H20_l2reg_0.0_a_1.0_b_0.1_sg_True_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
# filename = 'antmaze-large-diverse-v0_5_l2reg_0.0_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
# filename = 'antmaze-large-diverse-v0_5_l2reg_0.0_a_1.0_b_1.0_sg_False_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best.pth'
# filename = 'EM_model_antmaze-large-diverse-v0state_dec_mlp_H_40_l2reg_0.0_a_2.0_b_1.0_log_best.pth'
# filename = 'EM_model_antmaze-large-diverse-v0state_dec_mlp_init_state_dep_False_H_40_l2reg_0.0_a_2.0_b_1.0_log_best.pth'	
# filename = 'EM_model_antmaze-large-diverse-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_5.0_b_1.0_log_best.pth'
# filename = 'EM_model_antmaze-large-diverse-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_log_best.pth'
#filename = 'EM_model_antmaze-large-diverse-v0state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_2.0_b_1.0_log_best.pth'
filename = 'EM_model_maze2d-large-v1state_dec_mlp_init_state_dep_True_H_40_l2reg_0.0_a_1.0_b_1.0_per_el_sig_True_log_best.pth'

PATH = 'checkpoints/'+filename


if term_state_dependent_prior:
	skill_model = SkillModelTerminalStateDependentPrior(state_dim,a_dim,z_dim,h_dim,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,fixed_sig=fixed_sig).cuda()
elif state_dependent_prior:
	skill_model = SkillModelStateDependentPrior(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist,state_dec_stop_grad=state_dec_stop_grad,beta=beta,alpha=alpha,max_sig=max_sig,fixed_sig=fixed_sig,ent_pen=ent_pen,encoder_type=encoder_type,init_state_dependent=init_state_dependent).cuda()
else:
	skill_model = SkillModel(state_dim, a_dim, z_dim, h_dim, a_dist=a_dist).cuda()
checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

# initialize skill sequence
# skill_seq = torch.randn((1,skill_seq_len,z_dim), device=device, requires_grad=True)
# s0 = env.reset()
# initial_loc = s0[:2]
# env.env.reset_to_location
# print('s0: ', s0)
# s0_torch = torch.tensor(s0,dtype=torch.float32).cuda().reshape(1,1,-1)

s0_torch = torch.cat([torch.tensor(env.reset(),dtype=torch.float32).cuda().reshape(1,1,-1) for _ in range(batch_size)])

# z_mean,z_sig = skill_model.prior(s0_torch)
# print('z_mean: ', z_mean)
# print('z_sig: ', z_sig)
# skill_seq = z_mean.detach() + z_sig.detach() * torch.randn((1,skill_seq_len,z_dim), device=device)
# skill_seq = torch.randn((1,skill_seq_len,z_dim), device=device)
skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
print('skill_seq.shape: ', skill_seq.shape)
skill_seq.requires_grad = True

# s0_torch = torch.stack(batch_size*[torch.tensor(s0,dtype=torch.float32).cuda().unsqueeze(0)])

# s0 = torch.zeros((batch_size,1,state_dim), device=device)

# s0_torch = torch.stack([env.reset_to_location(initial_loc)])
# initialize optimizer for skill sequence
# determine waypoints
goal_state = np.array(env.get_target())#random.choice(data['observations'])
print('goal_state: ', goal_state)
# env.set_target(goal_state[:2])
goal_seq = torch.tensor(goal_state, device=device).reshape(1,1,-1)
#goal_seq = 2*torch.rand((1,skill_seq_len,state_dim), device=device) - 1

# experiment = Experiment(api_key = 'yQQo8E8TOCWYiVSruS7nxHaB5', project_name = 'skill-learning', workspace="anirudh-27")
# experiment.add_tag('Skill PLanning for '+env_name)
# experiment.log_parameters({'lr':lr,
# 							   'h_dim':h_dim,
# 							   'state_dependent_prior':state_dependent_prior,
# 							   'z_dim':z_dim,
# 				 						   'skill_seq_len':skill_seq_len,
# 			  				   'H':H,
# 			  				   'a_dim':a_dim,
# 			  				   'state_dim':state_dim,
# 			  				   'l2_reg':wd})
#experiment.log_metric('Goals', goal_seq)


# def make_gif(frame_folder):
#     frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
#     frame_one = frames[0]
#     frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
#                save_all=True, duration=100, loop=0)

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


def run_skills_iterative_replanning(env,model,goals,use_epsilon,replan_freq,variable_length,ep_num):
	
	s0 = env.reset()
	state = s0
	plt.scatter(s0[0],s0[1], label='Initial States')
	plt.scatter(goals[:,:,0].detach().cpu().numpy(),goals[:,:,1].detach().cpu().numpy(), label='Goals')
	plt.figure()
	# for i in range(skill_seq_len):
	# ipdb.set_trace()
	states = [s0]
	frames = []
	n=0
	timeout = False
	# success = True
	l = skill_seq_len
	while np.sum((state[:2] - goals.flatten().detach().cpu().numpy()[:2])**2) > 1.0:
	# for i in range(2):
		state_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32).cuda().reshape((1,1,-1))])
		
		if variable_length:
			cost_fn = lambda skill_seq,lengths: skill_model.get_expected_cost_variable_length(state_torch, skill_seq, lengths, goal_seq, use_epsilons=use_epsilon)
			skill_seq_mean = torch.zeros((skill_seq_len,z_dim),device=device)
			skill_seq_std  = torch.ones( (skill_seq_len,z_dim),device=device)
			p_lengths = (1/(skill_seq_len)) * torch.ones(skill_seq_len+1,device=device)
			p_lengths[0] = 0.0
		
		
			skill_seq_mean,skill_seq_std = cem_variable_length(skill_seq_mean,skill_seq_std,p_lengths,cost_fn,batch_size,keep_frac,n_iters,max_ep=max_ep,l2_pen=cem_l2_pen)

		else:
			cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(state_torch, skill_seq, goal_seq, use_epsilons=use_epsilon,length_cost=plan_length_cost)
			if n == 0:
				skill_seq_mean = torch.zeros((skill_seq_len,z_dim),device=device)
				skill_seq_std  = torch.ones( (skill_seq_len,z_dim),device=device)
			else:
				skill_seq_mean = torch.zeros((skill_seq_len,z_dim),device=device)
				skill_seq_std  = torch.ones( (skill_seq_len,z_dim),device=device)
			
				# skill_seq_mean = torch.cat([skill_seq_mean[1:,:],torch.zeros((1,z_dim),device=device)])
				# skill_seq_std  = torch.cat([skill_seq_std[1:,:], torch.ones((1,z_dim),device=device)])
		
			# 								           x_mean,        x_std,cost_fn,  pop_size,frac_keep,n_iters,l2_pen
			skill_seq_mean,skill_seq_std = cem(skill_seq_mean,skill_seq_std,cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)



		if skill_seq_mean.shape[0] == 0:

			print('OUT OF SKILLS!!!')
			# out_of_skills = True
			# break
		else:
			skill = skill_seq_mean[0,:].unsqueeze(0)

			
		if use_epsilon:
			mu_z, sigma_z = model.prior(torch.tensor(state,dtype=torch.float32).cuda().reshape(1,1,-1))
			z = mu_z + sigma_z*skill
		else:
			z = skill
		print('executing skill')
		for j in range(replan_freq):
		# for j in range(100):
			if render:
				frames.append(env.render(mode='rgb_array'))
			# env.render()
			# vid.capture_frame()
			action = model.decoder.ll_policy.numpy_policy(state,z)
			state,_,done,_ = env.step(action)
			# print('state: ', state)
			states.append(state)
			
			# skill_seq_states.append(state)
			# plt.scatter(state[0],state[1], label='Trajectory',c='b')
			if np.sum((state[:2] - goals.flatten().detach().cpu().numpy()[:2])**2) <= 1.0:
				break
			if done:
				print('DOOOOOOOOONE!!!!!!!!!!!!!!')
				print('state: ', state)
			# 	print('n: ',n)
				# break
		n += 1

		
	

		fig = plt.figure()
		# plt.imshow(background_img, extent = [-8,28,-8,28])
		plt.scatter(np.stack(states)[:,0],np.stack(states)[:,1])
		plt.scatter(goals.flatten().detach().cpu().numpy()[0],goals.flatten().detach().cpu().numpy()[1])
		plt.axis('equal')
		plt.savefig('ant_iterative_replanning_actual_states_niters'+str(n_iters)+'_l2pen_'+str(cem_l2_pen)+'.png')
	
		if n > max_replans*H/replan_freq:
			print('TIMEOUT!!!!!!!!!!!!!!')
			timeout = True
			break 
	


		# plt.savefig('ant_skills_iterative_replanning')
	# ipdb.set_trace()


	# save_frames_as_gif(frames)
	# for i,f in enumerate(frames):
	# 	plt.figure()
	# 	plt.imshow(f)
	# 	plt.savefig('ant'+str())
	env.close()
	# make_gif(frames,name='ant')
	if render or timeout:
		print('MAKING VIDEO!')
		# if timeout: 
		# 	print('making timout vid')
		# 	make_video(frames,name='failed_ant_'+str(j))
		# else:
		# 	make_video(frames,name='ant')
		make_video(frames,name='ant'+str(ep_num))
		# make_video(frames,name='yant')
		# make_video(frames,name='yant')


	states = np.stack(states)
	return states, np.min(np.sum((states[:,:2] - goals.flatten().detach().cpu().numpy()[:2])**2,axis=-1))


		


def run_skill_seq(skill_seq,env,s0,model,use_epsilon):
	'''
	'''
	# env.env.set_state(s0[:2],s0[2:])
	state = s0
	# s0_torch = torch.tensor(s0,dtype=torch.float32).cuda().reshape((1,1,-1))

	pred_states = []
	pred_sigs = []
	states = []
	# plt.figure()
	for i in range(skill_seq.shape[1]):
		# get the skill
		# z = skill_seq[:,i:i+1,:]
		if use_epsilon:
			mu_z, sigma_z = model.prior(torch.tensor(state,dtype=torch.float32).cuda().reshape(1,1,-1))
			z = mu_z + sigma_z*skill_seq[:,i:i+1,:]
		else:
			z = skill_seq[:,i:i+1,:]
		skill_seq_states = []
		state_torch = torch.tensor(state,dtype=torch.float32).cuda().reshape((1,1,-1))
		s_mean, s_sig = model.decoder.abstract_dynamics(state_torch,z)
		pred_state = s_mean.squeeze().detach().cpu().numpy()
		pred_sig = s_sig.squeeze().detach().cpu().numpy()
		pred_states.append(pred_state)
		pred_sigs.append(pred_sig)
		
		# run skill for H timesteps
		for j in range(H):
			action = model.decoder.ll_policy.numpy_policy(state,z)
			state,_,_,_ = env.step(action)
			skill_seq_states.append(state)
			#env.render()
			#plt.scatter(state[0],state[1], label='Trajectory',c='b')
		states.append(skill_seq_states)
		# plt.scatter(state[0],state[1], label='Term state',c='r')
		# plt.scatter(pred_state[0],pred_state[1], label='Pred states',c='g')
		# plt.errorbar(pred_state[0],pred_state[1],xerr=pred_sig[0],yerr=pred_sig[1],c='g')
		

	states = np.stack(states)
	goals = goal_seq.detach().cpu().numpy()
	goals = np.stack(goals)
	pred_states = np.stack(pred_states)
	pred_sigs = np.stack(pred_sigs)



	# plt.scatter(s0[0],s0[1], label='Initial States')
	# plt.scatter(goals[:,:,0],goals[:,:,1], label='Goals')
	
	# plt.axis('square')
	# plt.xlim([-1,38])
	# plt.ylim([-1,30])


	# if not state_dependent_prior:
	# 	plt.title('Planned Skills (No State Dependent Prior)')
	# 	plt.savefig('Skill_planning_H'+str(H)+'_sdp_'+'false'+'.png')
	# else:
	# 	plt.title('Planned skills (State Dependent Prior)')
	# 	plt.savefig('Skill_planning_H'+str(H)+'.png')

	# print('SAVED FIG!')

	return state,states



# min_dists_list = []
# for i in range(100):
# 	states,min_dist = run_skills_iterative_replanning(env,skill_model,goal_seq,use_epsilon,replan_freq,variable_length,i)
# 	min_dists_list.append(min_dist)
# 	print('min_dists_list: ',min_dists_list)
# 	np.save('min_dists_list_n_iters'+str(n_iters)+'_l2pen_'+str(cem_l2_pen),min_dists_list)


execute_n_skills = 1

min_dists_list = []
for j in range(1000):
	state = env.reset()
	goal_loc = goal_state[:2]
	min_dist = 10**10
	for i in range(max_replans):
		s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))])
		cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(s_torch, skill_seq, goal_seq, var_pen = var_pen)
		skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
		skill_seq = skill_seq[:execute_n_skills,:]
		skill_seq = skill_seq.unsqueeze(0)	
		skill_seq = convert_epsilon_to_z(skill_seq,s_torch[:1,:,:],skill_model)
		state,states = run_skill_seq(skill_seq,env,state,skill_model,use_epsilon=False)
		# print('states.shape: ', states.shape)
		#print(i)
		dists = np.sum((states[0,:,:2] - goal_loc)**2,axis=-1)

		if np.min(dists) < min_dist:
			min_dist = np.min(dists)

		if min_dist < 0.5:
			break
	
	min_dists_list.append(min_dist)
	np.save('min_dists_list_'+filename, min_dists_list)
	# print('min_dists_list: ', min_dists_list)
	n_success = np.sum(np.array(min_dists_list) <= 0.5)
	n_tot = len(min_dists_list)

	ci = proportion_confint(n_success,n_tot)
	print('ci: ', ci)
	print('mean: ',n_success/n_tot)
	print('n_tot: ',n_tot)



# s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))])
# cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(s_torch, skill_seq, goal_seq, var_pen = var_pen)
# skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
# skill_seq = skill_seq[:5,:]
# skill_seq = skill_seq.unsqueeze(0)		
# skill_seq = convert_epsilon_to_z(skill_seq,s0_torch[:1,:,:],skill_model)
# state = run_skill_seq(skill_seq,env,s0,skill_model,use_epsilon=False)

# s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))])
# cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(s_torch, skill_seq, goal_seq, var_pen = var_pen)
# skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
# skill_seq = skill_seq[:5,:]
# skill_seq = skill_seq.unsqueeze(0)		
# skill_seq = convert_epsilon_to_z(skill_seq,s0_torch[:1,:,:],skill_model)
# state = run_skill_seq(skill_seq,env,s0,skill_model,use_epsilon=False)

# s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))])
# cost_fn = lambda skill_seq: skill_model.get_expected_cost_for_cem(s_torch, skill_seq, goal_seq, var_pen = var_pen)
# skill_seq,_ = cem(torch.zeros((skill_seq_len,z_dim),device=device),torch.ones((skill_seq_len,z_dim),device=device),cost_fn,batch_size,keep_frac,n_iters,l2_pen=cem_l2_pen)
# skill_seq = skill_seq[:5,:]
# skill_seq = skill_seq.unsqueeze(0)		
# skill_seq = convert_epsilon_to_z(skill_seq,s0_torch[:1,:,:],skill_model)
# state = run_skill_seq(skill_seq,env,s0,skill_model,use_epsilon=False)

# plt.figure()
# plt.scatter(states[:,0],states[:,1])
# plt.scatter(goal_state[0],goal_state[1])
# plt.axis('equal')
# print('SAVING!')
# plt.savefig('ant_iterative_replanning_actual_states_n_iters'+str(n_iters)+'_l2pen_'+str(cem_l2_pen))



######## FOR RUNNING MANNY TRIALS OF ITERATIVE REPLANNING ########


# success_list = []
# for i in range(1):
# 	states,success = run_skills_iterative_replanning(env,skill_model,goal_seq)
# 	success_list.append(success)
# 	np.save('success_list',success_list)
# print('success_list: ', success_list)
# print('np.mean(success_list): ', np.mean(success_list))

