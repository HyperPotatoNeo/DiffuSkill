from argparse import ArgumentParser
import os
import pickle
import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.skill_model import SkillModel
from utils.utils import get_dataset, StateActionReturnTerminalsDataset

def collect_data(args):
    if 'antmaze' in args.env:
        state_dim = 29 + args.append_goals * 2
        a_dim = 8
    elif 'kitchen' in args.env:
        state_dim = 60
        a_dim = 9
    elif 'maze' in args.env:
        state_dim = 4 + args.append_goals * 2
        a_dim = 2
    elif 'Breakout' in args.env:
        state_dim = 64
        a_dim = 4
    elif 'pen' in args.env:
        state_dim = 45
        a_dim = 24
    elif 'door' in args.env:
        state_dim = 39
        a_dim = 28
    elif 'hammer' in args.env:
        state_dim = 46
        a_dim = 26
    elif 'relocate' in args.env:
        state_dim = 39
        a_dim = 30
    else:
        raise NotImplementedError

    skill_model_path = os.path.join(args.checkpoint_dir, args.skill_model_filename)

    checkpoint = torch.load(skill_model_path)

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
                             env_name=args.env
                             ).to(args.device)
    skill_model.load_state_dict(checkpoint['model_state_dict'])
    skill_model.eval()

    if args.do_diffusion:
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


    if 'atari' in args.env:
        obss, actions, rewards, done_idxs, timesteps = get_dataset(args.env, args.horizon, args.stride, 0.0, args.append_goals)

        inputs_train = StateActionReturnTerminalsDataset(obss, args.horizon * 3, actions, done_idxs, rewards, timesteps)
        train_loader = DataLoader(inputs_train,
                                    batch_size=args.batch_size,
                                    num_workers=0)
    else:
        dataset = get_dataset(args.env, args.horizon, args.stride, 0.0, args.append_goals, get_rewards=True, cum_rewards=args.cum_rewards)

        obs_chunks_train = dataset['observations_train']
        action_chunks_train = dataset['actions_train']
        rewards_chunks_train = dataset['rewards_train']

        if not 'maze' in args.env and not 'kitchen' in args.env and not 'pen' in args.env and not 'hammer' in args.env and not 'relocate' in args.env and not 'door' in args.env:
            terminals_chunks_train = dataset['terminals_train']
            inputs_train = torch.cat([obs_chunks_train, action_chunks_train, rewards_chunks_train, terminals_chunks_train], dim=-1)
        else:
            inputs_train = torch.cat([obs_chunks_train, action_chunks_train, rewards_chunks_train], dim=-1)

        train_loader = DataLoader(
            inputs_train,
            batch_size=args.batch_size,
            num_workers=0)

    states_gt = np.zeros((len(inputs_train), state_dim+2*args.append_goals))
    latent_gt = np.zeros((len(inputs_train), args.z_dim))
    if args.save_z_dist:
        latent_std_gt = np.zeros((len(inputs_train), args.z_dim))
    sT_gt = np.zeros((len(inputs_train), state_dim))
    rewards_gt = np.zeros((len(inputs_train), 1))

    if args.do_diffusion:
        diffusion_latents_gt = np.zeros((len(inputs_train), args.num_diffusion_samples, args.z_dim))
    else:
        prior_latents_gt = np.zeros((len(inputs_train), args.num_prior_samples, args.z_dim))

    if not 'maze' in args.env and not 'kitchen' in args.env:# and not 'atari' in args.env:
        terminals_gt = np.zeros((len(inputs_train), 1))
    gamma_array = np.power(args.gamma, np.arange(args.horizon))

    for batch_id, data in enumerate(tqdm(train_loader)):
        if 'atari' not in args.env:
            data = data.to(args.device)
            states = data[:, :, :skill_model.state_dim]
            actions = data[:, :, skill_model.state_dim+2*args.append_goals:skill_model.state_dim+2*args.append_goals+a_dim]
            if not 'maze' in args.env and not 'kitchen' in args.env and not 'atari' in args.env:
                rewards = data[:, :, skill_model.state_dim+2*args.append_goals+a_dim:skill_model.state_dim+2*args.append_goals+a_dim+1]
                terminals = data[:, :, skill_model.state_dim+2*args.append_goals+a_dim+1:]
            else:
                rewards = data[:, :, skill_model.state_dim+2*args.append_goals+a_dim:]

        else:
            states = data[0].to(args.device)
            actions = data[1].to(args.device)
            rewards = data[2].to(args.device)
            terminals = data[4].to(args.device)
            with torch.no_grad():
                states = skill_model.image_encoder(states)

        start_idx = batch_id * args.batch_size
        end_idx = start_idx + args.batch_size
        states_gt[start_idx : end_idx] = states[:, 0].cpu().numpy()
        sT_gt[start_idx: end_idx] = states[:, -1].cpu().numpy()
        rewards_gt[start_idx: end_idx, 0] = np.sum(rewards.cpu().numpy()[:,:,0]*gamma_array, axis=1)
        if not 'maze' in args.env and not 'kitchen' in args.env and not 'atari' in args.env and not 'pen' in args.env and not 'relocate' in args.env and not 'door' in args.env and not 'hammer' in args.env:
            terminals_gt[start_idx: end_idx] = np.sum(terminals.cpu().numpy(), axis=1)
        elif 'atari' in args.env:
            terminals_gt[start_idx: end_idx] = terminals.cpu().numpy()

        if not args.do_diffusion:
            with torch.no_grad():
                prior_latent_mean, prior_latent_std = skill_model.prior(states[:, -1, :skill_model.state_dim])
                prior_latent_mean = prior_latent_mean.repeat_interleave(args.num_prior_samples, 0)
                prior_latent_std = prior_latent_std.repeat_interleave(args.num_prior_samples, 0)

                prior_latents_gt[start_idx : end_idx] = torch.stack(torch.distributions.normal.Normal(prior_latent_mean, prior_latent_std).sample().chunk(states.shape[0])).cpu().numpy()
        else:
            diffusion_state = states[:, -1, :skill_model.state_dim].repeat_interleave(args.num_diffusion_samples, 0)
            with torch.no_grad():
                diffusion_latents_gt[start_idx : end_idx] = torch.stack(diffusion_model.sample_extra(diffusion_state, predict_noise=args.predict_noise, extra_steps=args.extra_steps).chunk(states.shape[0])).cpu().numpy()

        if 'atari' in args.env:
            output, output_std = skill_model.encoder(states, torch.nn.functional.one_hot(torch.squeeze(actions,dim=2), num_classes=a_dim))
        else:
            output, output_std = skill_model.encoder(states, actions)

        latent_gt[start_idx : end_idx] = output.detach().cpu().numpy().squeeze(1)
        if args.save_z_dist:
            latent_std_gt[start_idx : end_idx] = output_std.detach().cpu().numpy().squeeze(1)

    DATA_DIR = 'data/'

    np.save(DATA_DIR + args.skill_model_filename[:-4] + '_states.npy', states_gt)
    np.save(DATA_DIR + args.skill_model_filename[:-4] + '_latents.npy', latent_gt)
    np.save(DATA_DIR + args.skill_model_filename[:-4] + '_sT.npy', sT_gt)
    np.save(DATA_DIR + args.skill_model_filename[:-4] + '_rewards.npy', rewards_gt)
    if args.do_diffusion:
        np.save(DATA_DIR + args.skill_model_filename[:-4] + '_sample_latents.npy', diffusion_latents_gt)
    else:
        np.save(DATA_DIR + args.skill_model_filename[:-4] + '_prior_latents.npy', prior_latents_gt)
    if args.save_z_dist:
        np.save(DATA_DIR + args.skill_model_filename[:-4] + '_latents_std.npy', latent_std_gt)
    if not 'maze' in args.env and not 'kitchen' in args.env:# and not 'atari' in args.env:
        np.save(DATA_DIR + args.skill_model_filename[:-4] + '_terminals.npy', terminals_gt)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--append_goals', type=int, default=0)
    parser.add_argument('--save_z_dist', type=int, default=1)
    parser.add_argument('--cum_rewards', type=int, default=0)

    parser.add_argument('--do_diffusion', type=int, default=1)
    parser.add_argument('--num_diffusion_samples', type=int, default=1000)
    parser.add_argument('--num_prior_samples', type=int, default=1000)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--extra_steps', type=int, default=5)
    parser.add_argument('--predict_noise', type=int, default=0)

    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='autoregressive')
    parser.add_argument('--per_element_sigma', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=16)

    args = parser.parse_args()

    collect_data(args)
