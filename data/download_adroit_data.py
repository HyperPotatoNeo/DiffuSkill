import gym
import numpy as np

import collections
import pickle

import d4rl
import pickle

datasets = []
env_name = ''

for dataset_type in ['pen-human', 'hammer-human', 'door-human', 'relocate-human']:
	name = f'{dataset_type}-v1'
	env = gym.make(name)
	dataset = env.get_dataset()
	with open(name+".pkl","wb") as f:
		pickle.dump(dataset,f)