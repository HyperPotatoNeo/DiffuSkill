from itertools import product

from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

EXTRA_DIFFUSION_STEPS = [0, 2, 4, 8, 16, 32]
GUIDE_WEIGHTS = [0.0, 4.0, 8.0]

device = "cuda"
n_hidden = 512
n_T = 50
net_type = "transformer"


def test():
    # get datasets set up
    dataset = PriorDataset(
        DATASET_PATH, train_or_test="train", train_prop=0.90
    )

    mean = dataset.state_mean
    stddev = dataset.state_stddev

    x_shape = dataset.state_all.shape[1:]
    y_dim = dataset.latent_all.shape[1]

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

    model.eval()

    extra_diffusion_steps = EXTRA_DIFFUSION_STEPS
    use_kdes = [False, True] if exp_name == "diffusion" else [False]
    guide_weight_list = GUIDE_WEIGHTS if exp_name == "cfg" else [None]

    x_eval = (
        torch.Tensor(torch_data_train.state_all[idx])
        .type(torch.FloatTensor)
        .to(device)
    )

    for j in range(6 if not use_kde else 300):
        x_eval_ = x_eval.repeat(50, 1, 1, 1)
        if exp_name == "cfg":
            model.guide_w = guide_weight

        if extra_diffusion_step == 0:
            y_pred_ = (
                model.sample(x_eval_, extract_embedding=True)
                .detach()
                .cpu()
                .numpy()
            )

            if use_kde:
                # kde
                torch_obs_many = x_eval_
                action_pred_many = model.sample(torch_obs_many).cpu().numpy()
                # fit kde to the sampled actions
                kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(action_pred_many)
                # choose the max likelihood one
                log_density = kde.score_samples(action_pred_many)
                idx = np.argmax(log_density)
                y_pred_ = action_pred_many[idx][None, :]
        else:
            y_pred_ = model.sample_extra(x_eval_, extra_steps=extra_diffusion_step).detach().cpu().numpy()


if __name__ == '__main__':
    test()