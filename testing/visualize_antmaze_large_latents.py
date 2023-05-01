from argparse import ArgumentParser
import os

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import pickle
matplotlib.use('TkAgg')

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.skill_model import SkillModel
from utils.utils import get_dataset

from sklearn.cluster import KMeans
import umap

def visualize_kitchen_mixed_latents(args):
    dataset_file = 'data/antmaze-large-diverse-v2.pkl'
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    num_trajectories = np.where(dataset['timeouts'])[0].shape[0]
    assert num_trajectories == 999, 'Dataset has changed. Review the dataset extraction'

    print('Total trajectories: ', num_trajectories)

    observations = []
    actions = []

    tasks = np.zeros(num_trajectories)

    for traj_idx in range(num_trajectories):
        start_idx = traj_idx * 1001
        end_idx = (traj_idx + 1) * 1001

        obs = dataset['observations'][start_idx : end_idx]
        act = dataset['actions'][start_idx : end_idx]

        observations.append(torch.tensor(obs[:args.horizon], dtype=torch.float32))
        actions.append(torch.tensor(act[:args.horizon], dtype=torch.float32))

    observations = torch.stack(observations)
    actions = torch.stack(actions)

    state_dim = 29
    a_dim = 8

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

    with torch.no_grad():
        gt_latents, _ = skill_model.encoder(observations.to(args.device), actions.to(args.device))
        gt_latents = gt_latents.squeeze(1)

    umap_fit = umap.UMAP() #TSNE()
    gt_umap_2d = umap_fit.fit_transform(gt_latents.cpu().numpy())

    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(gt_umap_2d)
    print('Ground truth distribution: ', np.where(labels == 0)[0].shape[0], np.where(labels == 1)[0].shape[0], np.where(labels == 2)[0].shape[0])

    zero_shape = np.where(labels == 0)[0].shape[0]
    one_shape = np.where(labels == 1)[0].shape[0]
    two_shape = np.where(labels == 2)[0].shape[0]

    colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']

    plt.scatter(gt_umap_2d[:, 0], gt_umap_2d[:, 1], c=[colors[label] for label in labels])
    plt.show()

    diffusion_states = observations[:, 0].to(args.device).repeat_interleave(args.num_diffusion_samples, 0)
    with torch.no_grad():
        diffusion_latents = diffusion_model.sample_extra(diffusion_states, predict_noise=args.predict_noise, extra_steps=args.extra_steps)

    diffusion_umap_2d = umap_fit.transform(diffusion_latents.cpu().numpy())

    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(diffusion_umap_2d)
    print('Diffusion distribution: ', np.where(labels == 0)[0].shape[0], np.where(labels == 1)[0].shape[0], np.where(labels == 2)[0].shape[0])

    plt.scatter(diffusion_umap_2d[:, 0], diffusion_umap_2d[:, 1], c=[colors[label] for label in labels])
    plt.show()

    with torch.no_grad():
        prior_latents, _ = skill_model.prior(observations[:, 0:1].to(args.device))
        prior_latents = prior_latents.squeeze(1)

    prior_umap_2d = umap_fit.transform(prior_latents.cpu().numpy())

    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(prior_umap_2d)

    plt.scatter(prior_umap_2d[:, 0], prior_umap_2d[:, 1], c=[colors[label] for label in labels])
    plt.show()


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--num_diffusion_samples', type=int, default=1)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--extra_steps', type=int, default=5)
    parser.add_argument('--predict_noise', type=int, default=0)

    parser.add_argument('--horizon', type=int, default=20)
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

    visualize_kitchen_mixed_latents(args)
