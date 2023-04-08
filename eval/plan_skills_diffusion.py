from argparse import ArgumentParser

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


def eval_func(envs,
              state_dim,
              num_evals,
              num_parallel_envs,
              extra_steps,
              planning_depth,
              render):

    assert num_evals % num_parallel_envs == 0
    num_evals = num_evals // num_parallel_envs

    for _ in range(num_evals):
        state_0 = torch.zeros((num_parallel_envs, state_dim))
        goal_state = torch.zeros((num_parallel_envs, 2))

        for env_idx in envs:
            state_0[env_idx] = torch.from_numpy(envs[env_idx].reset(), dtype=torch.float32)
            goal_state[env_idx] = envs[env_idx].target_goal

        for env_step in range(1001):
            state = state_0.repeat_interleave(num_diffusion_samples, 0)

            latent_0 = diffusion_model.sample_extra(state, extra_steps=extra_steps)
            state = self.decoder.abstract_dynamics(state, latent_0)

            for depth in range(planning_depth):
                latent = diffusion_model.sample_extra(state, extra_steps=extra_steps)
                state = self.decoder.abstract_dynamics(state, latent)

            best_latent = torch.zeros((num_parallel_envs, latent_0.shape[1]))

            for env_idx in envs:
                start_idx = env_idx * num_diffusion_samples
                end_idx = start_idx + num_diffusion_samples

                cost = np.linalg.norm(state[start_idx : end_idx][:2] - goal_state[env_idx], axis=1)
                best_latent[env_idx] = latent_0[start_idx + torch.argmin(cost)]

            action, _ = skill_model.decoder.ll_policy(state_0, best_latent)

            for env_idx in envs:
                new_state, _, _, _ = envs[env_idx].step(action.detach().cpu().numpy())
                state_0[env_idx] = torch.from_numpy(new_state)

            if render:
                envs[0].render()


def evaluate(args):
    env = gym.make(args.env)
    dataset = env.get_dataset()
    state_dim = dataset['observations'].shape[1]
    a_dim = dataset['actions'].shape[1]

    checkpoint = torch.load(args.skill_model_path)

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

    diffusion_nn_model = torch.load(args.diffusion_model_path).to(args.device)

    diffusion_model = Model_Cond_Diffusion(
        diffusion_nn_model,
        betas=(1e-4, 0.02),
        n_T=args.diffusion_steps,
        device=device,
        x_dim=diffusion_nn_model.xshape,
        y_dim=diffusion_nn_model.y_dim,
        drop_prob=None,
        guide_w=args.cfg_weight,
    )

    envs = [gym.make(args.env) for _ in range(args.num_parallel_envs)]

    eval_func(envs,
              state_dim,
              args.num_evals,
              args.num_parallel_envs,
              args.extra_steps,
              args.planning_depth,
              args.render,
              )


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_evals', type=int, default=1)
    parser.add_argument('--num_parallel_envs', type=int, default=1)
    parser.add_argument('--skill_seq_len', type=int, default=1)
    parser.add_argument('--skill_model_path', type=str)
    parser.add_argument('--diffusion_model_path', type=str)

    parser.add_argument('--num_diffusion_samples', type=int, 50)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--planning_depth', type=int, default=3)
    parser.add_argument('--extra_steps', type=int, default=4)

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

    evaluate(args)