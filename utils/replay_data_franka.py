from argparse import ArgumentParser
import time

import gym

from utils import get_dataset

import gym
import d4rl

def replay_data(args):
    dataset = get_dataset(args.env, 1, args.stride, 0.0, get_rewards=True)

    state_dim = dataset['observations_train'].shape[-1]
    a_dim = dataset['actions_train'].shape[-1]

    env = gym.make(args.env)

    for idx, (observation, action, rewards) in enumerate(zip(dataset['observations_train'], dataset['actions_train'], dataset['rewards_train'])):
        if idx % 5 == 0:
            env.set_state(observation[0, :30], observation[0, 30:-1])
            env.render()
        # time.sleep(0.1)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='kitchen-partial-v0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--horizon', type=int, default=40)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='autoregressive')
    parser.add_argument('--per_element_sigma', type=int, default=0)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=256)

    args = parser.parse_args()

    replay_data(args)
