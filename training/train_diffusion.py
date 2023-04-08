from argparse import ArgumentParser
import os

import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)

DATASET_PATH = "dataset"
SAVE_DATA_DIR = "output"  # for models/data

n_epoch = 100
lrate = 1e-4
device = "cuda"
n_hidden = 512
batch_size = 32
n_T = 50
net_type = "transformer"


class PriorDataset(Dataset):
    def __init__(
        self, DATASET_PATH, train_or_test="train", train_prop=0.90
    ):
        # just load it all into RAM
        self.state_all = np.load(os.path.join(DATASET_PATH, "states.npy"), allow_pickle=True)
        self.latent_all = np.load(os.path.join(DATASET_PATH, "latents.npy"), allow_pickle=True)
        n_train = int(self.state_all.shape[0] * train_prop)
        if train_or_test == "train":
            self.state_all = self.state_all[:n_train]
            self.latent_all = self.latent_all[:n_train]
        elif train_or_test == "test":
            self.state_all = self.state_all[n_train:]
            self.latent_all = self.latent_all[n_train:]
        else:
            raise NotImplementedError

        self.state_mean = self.state_all.mean(axis=0)
        self.state_std = self.state_all.stddev(axis=0)

        self.state_all = (self.state_all - self.state_mean) / self.state_std

    def __len__(self):
        return self.state_all.shape[0]

    def __getitem__(self, index):
        state = self.state_all[index]
        latent = self.latent_all[index]

        return (state, latent)


def train(n_epoch, lrate, device, n_hidden, batch_size, n_T, net_type):
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

    # Unpack experiment settings
    exp_name = 'diffusion'
    drop_prob = 0.0

    # get datasets set up
    torch_data_train = PriorDataset(
        DATASET_PATH, train_or_test="train", train_prop=0.90
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    torch_data_test = PriorDataset(
        DATASET_PATH, train_or_test="test", train_prop=0.90
    )
    dataload_test = DataLoader(
        torch_data_test, batch_size=batch_size, shuffle=True, num_workers=0
    )

    x_shape = torch_data_train.state_all.shape[1:]
    y_dim = torch_data_train.latent_all.shape[1]

    # EBM langevin requires gradient for sampling
    requires_grad_for_eval = False

    # create model
    nn_model = Model_mlp(
        x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type
    ).to(device)
    model = Model_Cond_Diffusion(
        nn_model,
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        x_dim=x_shape,
        y_dim=y_dim,
        drop_prob=drop_prob,
        guide_w=0.0,
    )

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lrate)
    best_test_loss = 10000000

    for ep in tqdm(range(n_epoch), desc="Epoch"):
        results_ep = [ep]
        model.train()

        # lrate decay
        optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

        # train loop
        model.train()
        pbar = tqdm(dataload_train)
        loss_ep, n_batch = 0, 0

        for x_batch, y_batch in pbar:
            x_batch = x_batch.type(torch.FloatTensor).to(device)
            y_batch = y_batch.type(torch.FloatTensor).to(device)
            loss = model.loss_on_batch(x_batch, y_batch)
            optim.zero_grad()
            loss.backward()
            loss_ep += loss.detach().item()
            n_batch += 1
            pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
            optim.step()

            torch.save(model, 'diffusion_prior.pt')

        results_ep.append(loss_ep / n_batch)

        # test loop
        model.eval()
        pbar = tqdm(dataload_test)
        loss_ep, n_batch = 0, 0

        for x_batch, y_batch in pbar:
            x_batch = x_batch.type(torch.FloatTensor).to(device)
            y_batch = y_batch.type(torch.FloatTensor).to(device)
            loss = model.loss_on_batch(x_batch, y_batch)
            loss_ep += loss.detach().item()
            n_batch += 1
            pbar.set_description(f"test loss: {loss_ep/n_batch:.4f}")
        if loss_ep < best_test_loss:
            best_test_loss = loss_ep
            skill_model.prior = model
            torch.save(model, 'diffusion_prior_best.pt')
            torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, args.skill_model_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--net_type', type=str, default='transformer')
    parser.add_argument('--n_hidden', type=int, default=512)


    parser.add_argument('--skill_seq_len', type=int, default=1)
    parser.add_argument('--skill_model_path', type=str)
    parser.add_argument('--diffusion_model_path', type=str)

    parser.add_argument('--n_T', type=int, default=50)
    parser.add_argument('--cfg_weight', type=float, default=0.0)

    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='autoregressive')
    parser.add_argument('--per_element_sigma', type=int, default=0)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=256)

    os.makedirs(SAVE_DATA_DIR, exist_ok=True)
    train(args.n_epoch, args.lrate, device, args.n_hidden, args.batch_size, args.n_T, args.net_type, args)
