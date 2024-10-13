from configs.config import config
import os

from utils import *
from trainer.utils import restore_checkpoint
import torch
import torch.nn as nn
import scipy.io
import sys

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet_video import UNetVideoModel
from models.ema import ExponentialMovingAverage
from sde import VESDE, VPSDE
import numpy as np
from einops import rearrange


def sample():
    config.cuda = config.gpu is not None
    if config.cuda:
        # torch.cuda.set_device(config.gpu)
        device = 'cuda'
    else:
        device = 'cpu'
    dtype_torch = getattr(torch, config.dtype)

    config.version = 'kolmogorov_v0'
    config.sample_mode = 'experiment'
    config.num_samples_train = 100000
    config.batch_size = 32
    config.num_conditions = 2
    config.num_frames = 10
    config.num_components = 2
    config.image_size = 64
    config.dimension = config.image_size ** 2 * (config.num_frames + config.num_conditions)
    config.num_channels = config.num_components + config.num_conditions
    config.channel_modal = [config.num_frames*config.num_components, 1, 1]
    config.num_modals = len(config.channel_modal)
    net = UNetVideoModel(config.num_channels, model_channels=32, out_channels=config.num_channels, 
                            num_res_blocks=2, 
                            attention_resolutions=(8, 16), 
                            image_size=config.image_size, 
                            dropout=0.1, 
                            channel_mult=(1, 2, 4, 8),
                            conv_resample=True,
                            dims=2,
                            num_heads=1,
                            use_rpe_net=True)
    # net = UNet2dCond(in_channels=22, out_channels=22, init_features=32, z_dim=1, emb_dim=128, dropout=0.1)

    net = nn.DataParallel(net)
    # net.load_state_dict(torch.load(config.model_path + '/checkpoint_{}_{}.pth'.format(config.data, config.version),
    #                                map_location=device)['model'])
    # net = net.module
    net.to(device)
    ema = ExponentialMovingAverage(net.parameters(), decay=config.ema_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    state = dict(optimizer=optimizer, model=net, ema=ema, epoch=0, iteration=0)

    state = restore_checkpoint(config.checkpoint + f'/checkpoint_{config.data}_{config.version}.pth', state, device)
    net = state['model']
    optimizer = state['optimizer']
    ema = state['ema']

    data_raw = np.load('/home/lzy/projects_dir/generative_model/data/ns4/data_test_ns4.npy')
    rey = np.array([50, 125, 575, 1100, 1500])
    sigma = np.array([2, 4, 6, 8])
    _, z = np.meshgrid(np.linspace(0, 2 * np.pi, config.image_size, endpoint=False),
                        np.linspace(0, 2 * np.pi, config.image_size, endpoint=False), indexing='ij')
    x_test = data_raw.reshape(5, -1, 400, 64, 64, 2)
    mean, std = np.load(f'../data/{config.data}/scalar_{config.data}_{config.version}.npy')
    scalar = lambda x: (x-mean)/std
    scalar_inv = lambda x: x*std+mean

    # load test data
    samples = []
    seed_samples = 1
    rnd = np.random.RandomState(seed_samples)
    num_re = len(x_test)
    num_k = len(x_test[0])
    s_re, s_k = 4, 0

    config.overlap = 2
    config.num_steps = 20
    nf = config.num_frames
    ns = config.num_steps
    ol = config.overlap
    b = int(ns // (nf - ol) + 1)  # the number of samples that need to generate
    ns_real = b * (nf - ol) + ol

    config.channel_modal = None

    '''random samples'''
    i_batch, i_step = np.meshgrid(np.arange(len(x_test)*len(x_test[0])),
                                    np.arange(len(x_test[0, 0]) - ns_real))
    index = rnd.choice(i_batch.size, config.num_samples, replace=False)
    i_batch = i_batch.reshape(-1)[index]
    i_step = i_step.reshape(-1)[index]
    for i_batch, i_start, i_end in zip(i_batch, i_step, i_step + ns_real):
        i_re = int(i_batch // num_k)
        i_k = int(i_batch % num_k)
        single_sample = x_test[i_re, i_k, i_start:i_end].transpose(0, 3, 1, 2)
        f = np.sin(sigma[i_k] * z)[np.newaxis, np.newaxis].repeat(ns_real, axis=0)
        r = (1./rey[i_re]*100. * np.ones_like(f))
        samples.append(np.concatenate([single_sample, f, r], axis=1))
    samples = np.stack(samples, axis=0)

    # sample
    outer_loop = 200
    inner_loop = 5

    mask_ratio = 0.95
    noise_level = 0.
    num_obs = 400
    '''for Kolmogorov flow'''
    domain_size = 2*np.pi
    spacing = domain_size/config.image_size
    offsets = np.array([0., 0.])
    coords = np.random.rand(num_obs, 2) * domain_size
    sde = VESDE(config, sigma_min=config.beta_min, sigma_max=config.beta_max, N=outer_loop)
    # sde = VPSDE(config, beta_min=config.beta_min, beta_max=config.beta_max, N=outer_loop)
    net.eval()
    reap_fn = lambda x, num: torch.stack([x for _ in range(num)])

    with torch.no_grad():
        '''random points'''
        # mask = np.zeros([config.num_channels, config.image_size, config.image_size])
        # mask[:config.num_components] = mask_gen([1, config.image_size, config.image_size], mask_ratio, seed=seed_samples)
        # mask_torch = torch.from_numpy(mask).to(device)
        # transform_gen = transform = lambda x:x*mask[np.newaxis, np.newaxis] if isinstance(x, np.ndarray) else x*mask_torch[None, None]
        '''parameter conditioned'''
        transform_gen = transform = lambda x:x[:, :10, :config.num_components]
        
        samples_scale = samples.copy()
        samples_scale[:, :, :config.num_components] = scalar(samples_scale[:, :, :config.num_components])
        y = transform_gen(samples_scale+np.random.randn(*samples_scale.shape)*noise_level*std)
        print(y.shape)

        latent_mask = torch.stack([torch.ones([config.num_frames, 1, 1, 1]).float() for _ in range(len(y))]).to(device)
        obs_mask = torch.stack([torch.zeros([config.num_frames, 1, 1, 1]).float() for _ in range(len(y))]).to(device)
        frame_indices = torch.stack([torch.arange(config.num_frames) for _ in range(len(y))]).to(device)
        latent_mask = torch.ones([config.num_frames, 1, 1, 1]).float().to(device)
        obs_mask = torch.zeros([config.num_frames, 1, 1, 1]).float().to(device)
        frame_indices = torch.arange(config.num_frames).to(device)
        if isinstance(net.module, UNetVideoModel):
            net_fn = lambda x, t: net(x, x0=x, timesteps=t, latent_mask=reap_fn(latent_mask, len(x)), obs_mask=reap_fn(obs_mask, len(x)), frame_indices=reap_fn(frame_indices, len(x)))[0]
        else:
            def net_fn(x, t):
                h = net(torch.cat([rearrange(x[:, :, :config.num_components], 'b t c h w -> b (t c) h w'), x[:, 0, config.num_components:]], dim=1), t)
                h1 = rearrange(h[:, :-config.num_conditions], 'b (t c) h w -> b t c h w', c=config.num_components)
                return torch.cat([h1, h[:, None, -config.num_conditions:].repeat(1, len(h1[0]), 1, 1, 1)], dim=2)

        x_inpainted, x_generated = complete_video_pc_dps(config, net_fn, sde, y,
                                                            transform_gen,
                                                            alpha=0., beta=0, gamma1=0., gamma2=0., 
                                                            corrector=LangevinCorrector,
                                                            snr=0.128, save_sample_path=True,
                                                            n_steps=inner_loop,
                                                            device=device, dtype=config.dtype, eps=1e-12,
                                                            probability_flow=False, continuous=True,
                                                            data_scalar=scalar)
        x_pred = x_inpainted
        vor_pred = vor_cal_batch(np.concatenate([x_pred[0, :, 1:2], x_pred[0, :, :1]], axis=1), config.image_size, [0, domain_size])        # vor_cal_batch(np.concatenate([x_pred[0, :, 1:2], x_pred[0, :, :1]], axis=1), config.image_size, [0, domain_size])       scalar_inv(x_pred[0, :, 0])
        vor_ref = vor_cal_batch(np.concatenate([samples[0, :, 1:2], samples[0, :, :1]], axis=1), config.image_size, [0, domain_size])
        print(1.0/(x_pred[0, :, -1].mean()/100))


if __name__ == "__main__":
    sample()
