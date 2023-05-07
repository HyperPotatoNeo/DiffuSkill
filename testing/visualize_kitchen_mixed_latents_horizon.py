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

from sklearn.decomposition import PCA


def visualize_kitchen_mixed_latents(args):
    dataset_file = 'data/kitchen-mixed-v0.pkl'
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    num_trajectories = np.where(dataset['terminals'])[0].shape[0]

    print('Total trajectories: ', num_trajectories)

    terminals = np.where(dataset['terminals'])[0]
    terminals = np.append(-1, terminals)

    observations = []
    actions = []

    tasks = np.zeros(len(terminals) - 1, dtype=int)

    for traj_idx in range(1, len(terminals)):
        start_idx = terminals[traj_idx - 1] + 1
        end_idx = terminals[traj_idx] + 1

        obs = dataset['observations'][start_idx : end_idx]
        act = dataset['actions'][start_idx : end_idx]

        observations.append(torch.tensor(obs[:args.horizon], dtype=torch.float32))
        actions.append(torch.tensor(act[:args.horizon], dtype=torch.float32))

        # Microwave
        if dataset['observations'][start_idx + 70, 22] < -0.45:
            tasks[traj_idx - 1] = 1

        # Kettle
        elif abs(dataset['observations'][start_idx + 70, 24] - 0.7) < 0.2:
            tasks[traj_idx - 1] = 0

        else:
            tasks[traj_idx - 1] = 2

    observations = torch.stack(observations)
    actions = torch.stack(actions)

    state_dim = 60
    a_dim = 9

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

    with torch.no_grad():
        gt_latents, _ = skill_model.encoder(observations.to(args.device), actions.to(args.device))
        gt_latents = gt_latents.squeeze(1)

    umap_fit = PCA()
    gt_umap_2d = umap_fit.fit_transform(gt_latents.cpu().numpy())

    print('Ground truth distribution: ', np.where(tasks == 0)[0].shape[0], np.where(tasks == 1)[0].shape[0], np.where(tasks == 2)[0].shape[0])

    colors = ['#4285F4', '#DB4437', '#F4B400',] #Blue, Red, Yellow
    labels = ['Kettle', 'Microwave', 'Burner',]

    fig, ax = plt.subplots()

    # Set the background color to gray
    ax.set_facecolor('#EDEDED')

    # Customize grid lines to create a brick-like pattern
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.grid(which='major', linestyle='-', linewidth='1.5', color='white')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color='white')

    # plt.gca().set_aspect('equal')

    # Scatter plot
    ax.scatter(gt_umap_2d[:, 0], gt_umap_2d[:, 1], c=[colors[label] for label in tasks], zorder=2, alpha=0.6, s=50)

    # Set the x and y axis limits
    ax.set_xlim([-0.5, 1.0])
    ax.set_ylim([-0.4, 0.7])

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Remove border
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(handles, labels, fontsize=15)

    # Show the plot
    plt.show()

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='kitchen-partial-v0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--batch_size', type=int, default=64)

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
