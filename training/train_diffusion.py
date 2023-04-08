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

    for ep in tqdm(range(n_epoch), desc="Epoch"):
        results_ep = [ep]
        model.train()

        # lrate decay
        optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

        # train loop
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

            torch.save(model.state_dict(), 'diffusion_prior.pt')

        results_ep.append(loss_ep / n_batch)


if __name__ == "__main__":
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)
    train(n_epoch, lrate, device, n_hidden, batch_size, n_T, net_type)
