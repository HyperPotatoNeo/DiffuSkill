from argparse import ArgumentParser
import os

import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.skill_model import SkillModel
from utils.utils import get_dataset

ANTMAZE = plt.imread('img/maze-large.png')


def test_diffusion(args):
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

    if args.viz_diffusion:
        diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt')).to(args.device)

        diffusion_model = Model_Cond_Diffusion(
            diffusion_nn_model,
            betas=(1e-4, 0.02),
            n_T=args.diffusion_steps,
            device=args.device,
            x_dim=state_dim,
            y_dim=args.z_dim,
            drop_prob=None,
            guide_w=args.cfg_weight,
        )
        diffusion_model.eval()

    plt.imshow(ANTMAZE, extent=[-6, 42, -6, 30])

    for idx, (observation, action) in enumerate(zip(dataset['observations_train'], dataset['actions_train'])):
        print(idx)
        observation = observation.to(args.device).unsqueeze(0)
        action = action.to(args.device).unsqueeze(0)

        with torch.no_grad():
            reconstructed_observation, _, _, _, _, _, _ = skill_model(observation.repeat_interleave(10, 0), action.repeat_interleave(10, 0), True)

            if args.conditional_prior:
                latent_prior_mean, latent_prior_std = skill_model.prior(observation[:, 0:1])
                latent_prior = latent_prior_mean
            else:
                latent_prior = torch.distributions.Normal(torch.zeros((1, 1, args.z_dim)), torch.ones((1, 1, args.z_dim))).sample().to(args.device)

            if args.state_decoder_type == 'autoregressive':
                reconstructed_observation_prior, _ = skill_model.decoder.abstract_dynamics(observation[:, 0:1].repeat_interleave(10, 0), None, latent_prior.repeat_interleave(10, 0), evaluation=True)
            else:
                reconstructed_observation_prior, _ = skill_model.decoder.abstract_dynamics(observation[:, 0:1], latent_prior)
            reconstructed_observation_prior = reconstructed_observation_prior.squeeze(1)

            if args.viz_diffusion:
                state = observation[0, 0:1].repeat_interleave(args.num_diffusion_samples, 0)
                latent = diffusion_model.sample_extra(state, predict_noise=args.predict_noise, extra_steps=args.extra_steps)

                if args.state_decoder_type == 'autoregressive':
                    state, sigma = skill_model.decoder.abstract_dynamics(state.unsqueeze(1), None, latent.unsqueeze(1), evaluation=True)
                    state = state.squeeze(1)
                else:
                    state, _ = skill_model.decoder.abstract_dynamics(state.unsqueeze(1), latent.unsqueeze(1))
                    state = state.squeeze(1)



        if args.viz_diffusion:
            plt.scatter(state[:, 0].cpu().numpy(), state[:, 1].cpu().numpy(), color='blue', label='diffusion dist')

        # plt.scatter(observation[0, 0, 0].cpu().numpy(), observation[0, 0, 1].cpu().numpy(), color='green', label='current state')
        plt.scatter(reconstructed_observation[:, 0, 0].cpu().numpy(), reconstructed_observation[:, 0, 1].cpu().numpy(), color='yellow', label='vae dist')
        plt.scatter(reconstructed_observation_prior[:, 0].cpu().numpy(), reconstructed_observation_prior[:, 1].cpu().numpy(), color='red', label='prior dist')

        related_states = torch.linalg.norm(dataset['observations_train'][:, 0] - observation[:, 0].cpu(), axis=-1) < 3

        plt.scatter(dataset['observations_train'][related_states, -1, 0], dataset['observations_train'][related_states, -1, 1], color='pink', label='ground truth dist')

    plt.legend()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--num_diffusion_samples', type=int, default=50)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--extra_steps', type=int, default=4)
    parser.add_argument('--predict_noise', type=int, default=0)
    parser.add_argument('--viz_diffusion', type=int, default=1)

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

    test_diffusion(args)
