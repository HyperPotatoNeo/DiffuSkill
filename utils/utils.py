import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
import ipdb
import random


def reparameterize(mean, std):
	eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
	return mean + std*eps

def stable_weighted_log_sum_exp(x,w,sum_dim):
	a = torch.min(x)
	ipdb.set_trace()

	weighted_sum = torch.sum(w * torch.exp(x - a),sum_dim)

	return a + torch.log(weighted_sum)

def chunks(obs,actions,H,stride):
	'''
	obs is a N x 4 array
	goals is a N x 2 array
	H is length of chunck
	stride is how far we move between chunks.  So if stride=H, chunks are non-overlapping.  If stride < H, they overlap
	'''
	
	obs_chunks = []
	action_chunks = []
	N = obs.shape[0]
	for i in range(N//stride - H):
		start_ind = i*stride
		end_ind = start_ind + H
		
		obs_chunk = torch.tensor(obs[start_ind:end_ind,:],dtype=torch.float32)

		action_chunk = torch.tensor(actions[start_ind:end_ind,:],dtype=torch.float32)
		
		loc_deltas = obs_chunk[1:,:2] - obs_chunk[:-1,:2] #Franka or Maze2d
		
		norms = np.linalg.norm(loc_deltas,axis=-1)
		#USE VALUE FOR THRESHOLD CONDITION BASED ON ENVIRONMENT
		if np.all(norms <= 0.8): #Antmaze large 0.8 medium 0.67 / Franka 0.23 mixed/complete 0.25 partial / Maze2d 0.22
			obs_chunks.append(obs_chunk)
			action_chunks.append(action_chunk)
		else:
			pass

	print('len(obs_chunks): ',len(obs_chunks))
	print('len(action_chunks): ',len(action_chunks))
			
	return torch.stack(obs_chunks),torch.stack(action_chunks)


def get_dataset(env_name, horizon, stride, test_split=0.2):
    env = gym.make(env_name)
    dataset = env.get_dataset()

    if env_name == 'antmaze-large-diverse-v2':
        observations = []
        actions = []
        # rewards = []
        # goals = []

        num_trajectories = np.where(dataset['timeouts'])[0].shape[0]
        assert num_trajectories == 999, 'Dataset has changed. Review the dataset extraction'

        print('Total trajectories: ', num_trajectories)

        for traj_idx in range(num_trajectories):
            start_idx = traj_idx * 1001
            end_idx = (traj_idx + 1) * 1001

            obs = dataset['observations'][start_idx : end_idx]
            act = dataset['actions'][start_idx : end_idx]
            # reward = dataset['rewards'][start_idx : end_idx]
            # goal = dataset['infos/goal'][start_idx : end_idx]

            num_observations = obs.shape[0]

            for chunk_idx in range(num_observations // stride - horizon):
                chunk_start_idx = chunk_idx * stride
                chunk_end_idx = chunk_start_idx + horizon

                observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                # rewards.append(torch.tensor(reward[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                # goals.append(torch.tensor(goal[chunk_start_idx : chunk_end_idx], dtype=torch.float32))

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        # rewards = torch.stack(rewards)
        # goals = torch.stack(goals)

        num_samples = observations.shape[0]
        # print(num_samples)
        # assert num_samples == 960039, 'Dataset has changed. Review the dataset extraction'

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
        train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        # rewards_train = rewards[train_indices]
        # goals_train = goals[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        # rewards_test = rewards[test_indices]
        # goals_test = goals[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    # rewards_train=rewards_train,
                    # goals_train=goals_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    # rewards_test=rewards_test,
                    # goals_test=goals_test,
                    )

    else:
        raise NotImplementedError
