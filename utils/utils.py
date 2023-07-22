#import d4rl
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
import pickle
from utils.fixed_replay_buffer import FixedReplayBuffer

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


def create_dataset(num_buffers=50, num_steps=500000, game='Breakout', data_dir_prefix='data/dqn_replay/', trajectories_per_buffer=10):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0
    while len(obss) < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]
        print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + '/1/replay_logs',
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000)
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer
            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
                obss += [states]
                actions += [ac[0]]
                stepwise_returns += [ret[0]]
                if terminal[0]:
                    done_idxs += [len(obss)]
                    returns += [0]
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                returns[-1] += ret[0]
                i += 1
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0
                    i = transitions_per_buffer[buffer_num]
                    done = True
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)
            transitions_per_buffer[buffer_num] = i
        print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return obss, actions, stepwise_returns, done_idxs, timesteps


def get_dataset(env_name, horizon, stride, test_split=0.2, append_goals=False, get_rewards=False, separate_test_trajectories=False, cum_rewards=True):
    if 'atari' not in env_name:
        dataset_file = 'data/'+env_name+'.pkl'
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)

    observations = []
    actions = []
    terminals = []
    if get_rewards:
        rewards = []
    # goals = []

    if env_name == 'antmaze-large-diverse-v2' or env_name == 'antmaze-medium-diverse-v2':

        num_trajectories = np.where(dataset['timeouts'])[0].shape[0]
        assert num_trajectories == 999, 'Dataset has changed. Review the dataset extraction'

        if append_goals:
            dataset['observations'] = np.hstack([dataset['observations'],dataset['infos/goal']])
        print('Total trajectories: ', num_trajectories)

        for traj_idx in range(num_trajectories):
            start_idx = traj_idx * 1001
            end_idx = (traj_idx + 1) * 1001

            obs = dataset['observations'][start_idx : end_idx]
            act = dataset['actions'][start_idx : end_idx]
            if get_rewards:
                rew = np.expand_dims(dataset['rewards'][start_idx : end_idx],axis=1)
                
            # reward = dataset['rewards'][start_idx : end_idx]
            # goal = dataset['infos/goal'][start_idx : end_idx]

            num_observations = obs.shape[0]

            for chunk_idx in range(num_observations // stride - horizon):
                chunk_start_idx = chunk_idx * stride
                chunk_end_idx = chunk_start_idx + horizon

                observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                if get_rewards:
                    if np.sum(rew[chunk_start_idx : chunk_end_idx]>0):
                        rewards.append(torch.ones((chunk_end_idx-chunk_start_idx,1), dtype=torch.float32))
                        break
                    rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                # goals.append(torch.tensor(goal[chunk_start_idx : chunk_end_idx], dtype=torch.float32))

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        if get_rewards:
            rewards = torch.stack(rewards)
        # goals = torch.stack(goals)

        num_samples = observations.shape[0]
        # print(num_samples)
        # assert num_samples == 960039, 'Dataset has changed. Review the dataset extraction'

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        if get_rewards:
            rewards_train = rewards[train_indices]
        else:
            rewards_train = None
        # goals_train = goals[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        if get_rewards:
            rewards_test = rewards[test_indices]
        else:
            rewards_test = None
        # goals_test = goals[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    # goals_train=goals_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    # goals_test=goals_test,
                    )

    elif 'kitchen' in env_name:

        num_trajectories = np.where(dataset['terminals'])[0].shape[0]

        print('Total trajectories: ', num_trajectories)

        terminals = np.where(dataset['terminals'])[0]
        terminals = np.append(-1, terminals)

        for traj_idx in range(1, len(terminals)):
            start_idx = terminals[traj_idx - 1] + 1
            end_idx = terminals[traj_idx] + 1

            obs = dataset['observations'][start_idx : end_idx]
            act = dataset['actions'][start_idx : end_idx]
            rew = np.expand_dims(dataset['rewards'][start_idx : end_idx],axis=1)

            # reward = dataset['rewards'][start_idx : end_idx]
            # goal = dataset['infos/goal'][start_idx : end_idx]

            num_observations = obs.shape[0]

            for chunk_idx in range(num_observations // stride - horizon):
                chunk_start_idx = chunk_idx * stride
                chunk_end_idx = chunk_start_idx + horizon

                observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                if cum_rewards:
                    rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
                else:
                    rewards.append(torch.tensor(np.diff(rew[chunk_start_idx : chunk_end_idx], axis=0, prepend=rew[chunk_start_idx, 0]), dtype=torch.float32))
                # goals.append(torch.tensor(goal[chunk_start_idx : chunk_end_idx], dtype=torch.float32))

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)

        num_samples = observations.shape[0]

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        rewards_train = rewards[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        rewards_test = rewards[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    )

    elif 'maze2d' in env_name:

        if append_goals:
            dataset['observations'] = np.hstack([dataset['observations'], dataset['infos/goal']])

        obs = dataset['observations']
        act = dataset['actions']

        if get_rewards:
            rew = np.expand_dims(dataset['rewards'], axis=1)

        # reward = dataset['rewards'][start_idx : end_idx]
        # goal = dataset['infos/goal'][start_idx : end_idx]

        num_observations = obs.shape[0]

        for chunk_idx in range(num_observations // stride - horizon):
            chunk_start_idx = chunk_idx * stride
            chunk_end_idx = chunk_start_idx + horizon

            observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            if get_rewards:
                rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            # goals.append(torch.tensor(goal[chunk_start_idx : chunk_end_idx], dtype=torch.float32))

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        if get_rewards:
            rewards = torch.stack(rewards)
        # goals = torch.stack(goals)

        num_samples = observations.shape[0]

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        if get_rewards:
            rewards_train = rewards[train_indices]
        else:
            rewards_train = None
        # goals_train = goals[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        if get_rewards:
            rewards_test = rewards[test_indices]
        else:
            rewards_test = None
        # goals_test = goals[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    # goals_train=goals_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    # goals_test=goals_test,
                    )

    elif 'atari' in env_name:
        env_name = env_name[6:]
        obss, actions, rewards, done_idxs, timesteps = create_dataset(game=env_name)
        return obss, actions, rewards, done_idxs, timesteps

    else:
        obs = dataset['observations']
        act = dataset['actions']
        rew = np.expand_dims(dataset['rewards'],axis=1)
        dones = np.expand_dims(dataset['terminals'],axis=1)
        episode_step = 0
        chunk_idx = 0

        while chunk_idx < rew.shape[0]-horizon+1:
            chunk_start_idx = chunk_idx
            chunk_end_idx = chunk_start_idx + horizon

            observations.append(torch.tensor(obs[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            actions.append(torch.tensor(act[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            rewards.append(torch.tensor(rew[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            terminals.append(torch.tensor(dones[chunk_start_idx : chunk_end_idx], dtype=torch.float32))
            if np.sum(dones[chunk_start_idx : chunk_end_idx]>0):
                episode_step = 0
                chunk_idx += horizon
            elif(episode_step==1000-horizon):
                episode_step = 0
                chunk_idx += horizon
            else:
                episode_step += 1
                chunk_idx += 1

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        terminals = torch.stack(terminals)

        num_samples = observations.shape[0]

        print('Total data samples extracted: ', num_samples)
        num_test_samples = int(test_split * num_samples)

        if separate_test_trajectories:
            train_indices = np.arange(0, num_samples - num_test_samples)
            test_indices = np.arange(num_samples - num_test_samples, num_samples)
        else:
            test_indices = np.random.choice(np.arange(num_samples), num_test_samples, replace=False)
            train_indices = np.array(list(set(np.arange(num_samples)) - set(test_indices)))
        np.random.shuffle(train_indices)

        observations_train = observations[train_indices]
        actions_train = actions[train_indices]
        rewards_train = rewards[train_indices]
        terminals_train = terminals[train_indices]

        observations_test = observations[test_indices]
        actions_test = actions[test_indices]
        rewards_test = rewards[test_indices]
        terminals_test = terminals[test_indices]

        return dict(observations_train=observations_train,
                    actions_train=actions_train,
                    rewards_train=rewards_train,
                    terminals_train=terminals_train,
                    observations_test=observations_test,
                    actions_test=actions_test,
                    rewards_test=rewards_test,
                    terminals_test=terminals_test
                    )


class StateActionReturnDataset(torch.utils.data.Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rewards = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rewards, timesteps
