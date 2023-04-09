from argparse import ArgumentParser
import os

import numpy as np
import torch
import random
import gym
import d4rl
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.skill_model import SkillModel


def eval_func(diffusion_model,
              skill_model,
              envs,
              state_dim,
              state_mean,
              state_std,
              num_evals,
              num_parallel_envs,
              num_diffusion_samples,
              extra_steps,
              planning_depth,
              exec_horizon,
              predict_noise,
              render):

    with torch.no_grad():
        assert num_evals % num_parallel_envs == 0
        num_evals = num_evals // num_parallel_envs

        success_evals = 0

        for eval_step in range(num_evals):
            state_0 = torch.zeros((num_parallel_envs, state_dim)).to(args.device)
            goal_state = torch.zeros((num_parallel_envs, 2)).to(args.device)
            done = [False] * num_parallel_envs

            for env_idx in range(len(envs)):
                state_0[env_idx] = torch.from_numpy(envs[env_idx].reset())
                goal_state[env_idx][0] = envs[env_idx].target_goal[0]
                goal_state[env_idx][1] = envs[env_idx].target_goal[1]

            env_step = 0

            while env_step < 1000:
                state = state_0.repeat_interleave(num_diffusion_samples, 0)

                latent_0 = diffusion_model.sample_extra((state - state_mean) / state_std, predict_noise=predict_noise, extra_steps=extra_steps)

                state, _ = skill_model.decoder.abstract_dynamics(state, latent_0)

                for depth in range(1, planning_depth):
                    latent = diffusion_model.sample_extra((state - state_mean) / state_std, predict_noise=predict_noise, extra_steps=extra_steps)
                    state, _ = skill_model.decoder.abstract_dynamics(state, latent)

                best_latent = torch.zeros((num_parallel_envs, latent_0.shape[1])).to(args.device)

                for env_idx in range(len(envs)):
                    start_idx = env_idx * num_diffusion_samples
                    end_idx = start_idx + num_diffusion_samples

                    cost = torch.linalg.norm(state[start_idx : end_idx][:, :2] - goal_state[env_idx], axis=1)
                    best_latent[env_idx] = latent_0[start_idx + torch.argmin(cost)]

                for _ in range(exec_horizon):
                    for env_idx in range(len(envs)):
                        if not done[env_idx]:
                            action = skill_model.decoder.ll_policy.numpy_policy(state_0[env_idx], best_latent[env_idx])
                            new_state, reward, done[env_idx], _ = envs[env_idx].step(action)
                            success_evals += reward
                            state_0[env_idx] = torch.from_numpy(new_state)

                            if render and env_idx == 0:
                                envs[env_idx].render()

                    env_step += 1
                    if env_step > 1000:
                        break

            total_runs = (eval_step + 1) * num_parallel_envs
            print(f'Total successful evaluations: {success_evals} out of {total_runs} i.e. {success_evals / total_runs * 100}%')


def evaluate(args):
    env = gym.make(args.env)
    dataset = env.get_dataset()
    state_dim = dataset['observations'].shape[1]
    a_dim = dataset['actions'].shape[1]

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename))

    skill_model = SkillModel(state_dim,
                             a_dim,
                             args.z_dim,
                             args.h_dim,
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

    envs = [gym.make(args.env) for _ in range(args.num_parallel_envs)]

    state_all = np.load(os.path.join(args.dataset_dir, args.skill_model_filename[:-4] + "_states.npy"), allow_pickle=True)
    state_mean = torch.from_numpy(state_all.mean(axis=0)).to(args.device).float()
    state_std = torch.from_numpy(state_all.std(axis=0)).to(args.device).float()

    eval_func(diffusion_model,
              skill_model,
              envs,
              state_dim,
              state_mean,
              state_std,
              args.num_evals,
              args.num_parallel_envs,
              args.num_diffusion_samples,
              args.extra_steps,
              args.planning_depth,
              args.exec_horizon,
              args.predict_noise,
              args.render,
              )


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_evals', type=int, default=1)
    parser.add_argument('--num_parallel_envs', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--diffusion_model_filename', type=str)

    parser.add_argument('--num_diffusion_samples', type=int, default=50)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--planning_depth', type=int, default=3)
    parser.add_argument('--extra_steps', type=int, default=4)
    parser.add_argument('--predict_noise', type=int, default=0)
    parser.add_argument('--exec_horizon', type=int, default=40)

    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='autoregressive')
    parser.add_argument('--per_element_sigma', type=int, default=0)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=256)

    parser.add_argument('--render', type=int, default=1)

    args = parser.parse_args()

    evaluate(args)
