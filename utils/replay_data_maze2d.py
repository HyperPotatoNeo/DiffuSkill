from argparse import ArgumentParser
import os
import time

import gym
import d4rl

from utils import get_dataset

def replay_data(args):
    dataset = get_dataset(args.env, args.horizon, args.stride, 0.0, get_rewards=True)

    state_dim = dataset['observations_train'].shape[-1]
    a_dim = dataset['actions_train'].shape[-1]

    env = gym.make(args.env)

    for idx, (observation, action, reward) in enumerate(zip(dataset['observations_train'], dataset['actions_train'], dataset['rewards_train'])):
        env.set_state(observation[0, :2], observation[0, 2:])
        env.render()
        # print(reward[0])
        # time.sleep(0.1)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='maze2d-large-v1')
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)

    args = parser.parse_args()

    replay_data(args)
