from argparse import ArgumentParser
import os

import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from models.skill_model import SkillModel
from utils.utils import get_dataset

ANTMAZE = plt.imread('img/maze-large.png')

def test_vae(args):
    dataset = get_dataset(args.env, args.horizon, args.stride, 0.0)

    state_dim = dataset['observations_train'].shape[-1]
    a_dim = dataset['actions_train'].shape[-1]

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename))

    skill_model = SkillModel(state_dim,
                             a_dim,
                             args.z_dim,
                             args.h_dim,
                             args.horizon,
                             a_dist=args.a_dist,
                             beta=args.beta,
                             fixed_sig=None,
                             encoder_type=args.encoder_type,
                             state_decoder_type=args.state_decoder_type,
                             policy_decoder_type=args.policy_decoder_type,
                             per_element_sigma=args.per_element_sigma,
                             conditional_prior=args.conditional_prior,
                             ).to(args.device)
    skill_model.load_state_dict(checkpoint['model_state_dict'])
    skill_model.eval()

    for observation, action in zip(dataset['observations_train'], dataset['actions_train']):
        observation = observation.to(args.device).unsqueeze(0)
        action = action.to(args.device).unsqueeze(0)

        with torch.no_grad():
            reconstructed_observation, _, _, _, _, _ = skill_model(observation, action, True)

            if args.conditional_prior:
                latent_prior, _ = skill_model.prior(observation[:, 0:1])
            else:
                latent_prior = torch.distributions.Normal(torch.zeros((1, 1, args.z_dim)), torch.ones((1, 1, args.z_dim))).sample().to(args.device)

            if args.state_decoder_type == 'autoregressive':
                reconstructed_observation_prior, _ = skill_model.decoder.abstract_dynamics(observation[:, 0:1], None, latent_prior, evaluation=True)
                reconstructed_observation_prior = reconstructed_observation_prior.squeeze(1)
            else:
                reconstructed_observation_prior, _ = skill_model.decoder.abstract_dynamics(observation[:, 0:1], latent_prior)

        reconstructed_observation_prior = reconstructed_observation_prior.cpu().numpy()
        reconstructed_observation = reconstructed_observation.cpu().numpy()
        states = observation.cpu().numpy()

        plt.imshow(ANTMAZE, extent=[-6, 42, -6, 30])
        plt.scatter(reconstructed_observation_prior[:, 0], reconstructed_observation_prior[:, 1], color='red', label='prior')
        plt.scatter(reconstructed_observation[0, :, 0], reconstructed_observation[0, :, 1], color='yellow', label='decoder')
        plt.scatter(states[0, :, 0], states[0, :, 1], color='pink', label='ground truth trajectory')
        plt.scatter(states[0, 0, 0], states[0, 0, 1], color='blue', label='current state')
        plt.scatter(states[0, -1, 0], states[0, -1, 1], color='green', label='ground truth final')
        plt.legend()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--skill_model_filename', type=str)
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

    test_vae(args)
