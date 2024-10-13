import numpy as np
from torch.utils.data import Dataset
from einops import rearrange
import torch.nn.functional as F


def get_dataset(config, data, train=True):
    num_samples = config.num_samples_train if train else config.num_samples_val
    if len(data) >= 4:
        ind_train = int(config.train_split*len(data))
        data = data[:ind_train] if train else data[ind_train:]
    else:
        ind_train = int(config.train_split*len(data[0]))
        data = data[:, :ind_train] if train else data[:, ind_train:]
    is_condition = False if config.num_conditions == 0 else True
    if 'kse' in config.data.lower():
        return DatasetKSEVideo(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
    elif 'kol' in config.data.lower():
        return DatasetKolmogorov(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
    elif 'compns'  in config.data.lower():
        return DatasetCompNS(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif 'era5' in config.data.lower():
        return DatasetERA5(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif 'cylinder' in config.data.lower():
        return DatasetCylinder(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
    else:
        raise NotImplementedError('Unexpected type of data!')
    

def get_dataset_inverse(config, data, train=True):
    num_samples = config.num_samples_train if train else config.num_samples_val
    if len(data) >= 4:
        ind_train = int(config.train_split*len(data))
        data = data[:ind_train] if train else data[ind_train:]
    else:
        ind_train = int(config.train_split*len(data[0]))
        data = data[:, :ind_train] if train else data[:, ind_train:]
    is_condition = False if config.num_conditions == 0 else True
    if 'kse' in config.data.lower():
        return DatasetKSEInverse(data, num_samples, config.num_frames, degen_type=config.degen_type, scale=config.scale, is_scalar=config.is_scalar, is_condition=is_condition)
    elif 'kol' in config.data.lower():
        return DatasetKolInverse(data, num_samples, config.num_frames, degen_type=config.degen_type, scale=config.scale, is_scalar=config.is_scalar, is_condition=is_condition)
    else:
        raise NotImplementedError('Unexpected type of data!')


class DatasetKolmogorov(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        self.is_condition = is_condition
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        # the following is for Kolmogorov flow
        u0 = 1.0
        rey = np.linspace(100, 1050, 20, endpoint=True)
        sigma = np.linspace(2, 8, 7, endpoint=True)
        r, s = np.meshgrid(rey, sigma, indexing='ij')
        self.rey = r.reshape(-1)
        self.num_rey = len(self.rey)

        self.vis = u0 / self.rey
        self.f = [1. * np.sin(k * self.y) for k in s.reshape(-1)]

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        shift_x, shift_y = np.random.choice(self.num_h, 1)[0], np.random.choice(self.num_w, 1)[0]   # data aug.
        x = np.roll(x, (shift_x, shift_y), axis=(2, 3))     # data aug.
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        if self.is_condition:
            r = self.vis[i_b%self.num_rey]*100*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            f = self.f[i_b%self.num_rey][np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            return np.concatenate([x, f, r], axis=1), frame_indices, obs_mask, latent_mask       # B T C H W
        else:
            return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length
    

class DatasetCompNS(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        """data: list/tuple [dataset1, dataset2, ...], each of the dataset is of shape b*t*h*h*c"""
        data = data
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')

        mean, std = [], []
        for i in range(self.num_c):
            mean.append(data[..., i].mean())
            std.append(data[..., i].std())
        self.mean = np.array(mean)
        self.std = np.array(std)
        print(f'Statistics of Data (mean, std): {self.mean}, {self.std}')
        if is_scalar:
            self.data = (data-self.mean[np.newaxis, np.newaxis, np.newaxis, np.newaxis])/self.std[np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        shift_x, shift_y = np.random.choice(self.num_h, 1)[0], np.random.choice(self.num_w, 1)[0]   # data aug.
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        x = np.roll(x, (shift_x, shift_y), axis=(2, 3))     # data aug.
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length
    

class DatasetERA5(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length


class DatasetKSE(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        self.is_condition = is_condition
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        # the following is for Kolmogorov flow

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length
    

class DatasetKSEVideo(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        self.y = np.transpose(np.conj(np.arange(1, self.num_h+1))) / self.num_h * 2*np.pi
        # the following is for Kolmogorov flow
        self.is_condition = is_condition

        vis_min, vis_max, n_vis = 1, 5, 20
        self.vis = vis_min + (vis_max-vis_min) * np.arange(0, n_vis+1)/n_vis
        self.vis_scale = (self.vis-(vis_min+vis_max)/2)/(vis_max-vis_min)
        self.num_vis = len(self.vis)

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval]
        x = x.transpose(0, 2, 1)
        shift_x = np.random.choice(self.num_h, 1)[0]   # data aug.
        x = np.roll(x, shift_x, axis=2)     # data aug.
        latent_mask = np.ones([self.num_frames, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        if self.is_condition:
            r = self.vis_scale[i_b%self.num_vis]*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            return np.concatenate([x, r], axis=1), frame_indices, obs_mask, latent_mask       # B T C H
        else:
            return x, frame_indices, obs_mask, latent_mask 

    def __len__(self):
        return self.length


class DatasetCylinder(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: list/tuple [dataset1, dataset2, ...], each of the dataset is of shape b*t*h*h*c"""
        data = data
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        self.is_condition = is_condition
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        # the following is for cylinder flow
        self.vis = np.arange(0.003, 0., -0.0001).astype(dtype)
        self.u0 = 0.1
        # self.f = [np.zeros_like(y) for _ in range(len(self.vis))]
        self.rey = self.u0 / self.vis
        self.num_rey = len(self.rey)
        # the following is for Kolmogorov flow

        self.mean = np.mean(rearrange(data, 'b t h w c -> (b t h w) c'), axis=0)
        self.std = np.std(rearrange(data, 'b t h w c -> (b t h w) c'), axis=0)
        if is_scalar:
            self.data = (data-self.mean[np.newaxis, np.newaxis, np.newaxis, np.newaxis])/self.std[np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        if self.is_condition:
            r =  self.vis[i_b%self.num_rey]*1000*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            return np.concatenate([x, r], axis=1), frame_indices, obs_mask, latent_mask       # B T C H W
        else:
            return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length


class DatasetKSEInverse(Dataset):
    def __init__(self, data, length, num_frames=1, degen_type='fi', scale=64, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.y = np.transpose(np.conj(np.arange(1, self.num_h+1))) / self.num_h * 2*np.pi
        # the following is for Kolmogorov flow
        self.degen_type = degen_type
        self.scale = scale
        self.is_condition = is_condition
        self.num_frames = num_frames

        vis_min, vis_max, n_vis = 1, 5, 20
        self.vis = vis_min + (vis_max-vis_min) * np.arange(0, n_vis+1)/n_vis
        self.vis_scale = (self.vis-(vis_min+vis_max)/2)/(vis_max-vis_min)
        self.num_vis = len(self.vis)

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames]
        x = rearrange(x, 't h c -> (t c) h')
        shift_x = np.random.choice(self.num_h, 1)[0]   # data aug.
        x = np.roll(x, shift_x, axis=1)     # data aug.
        if 'fi' in self.degen_type:
            x_lr = x[:, ::self.scale]
        elif 'spec_real' in self.degen_type:
            x_lr = np.fft.fftshift(np.fft.fft(x[:, ::self.scale], axis=-1).real, axes=-1)
        r = self.vis_scale[i_b%self.num_vis]*np.ones_like(self.y)[np.newaxis]
        return x_lr, x, r, self.y[np.newaxis]       # B C H

    def __len__(self):
        return self.length


class DatasetKolInverse(Dataset):
    def __init__(self, data, length, num_frames=1, degen_type='fi', scale=8, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.is_condition = is_condition
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        x, y = [np.linspace(0, 2*np.pi, self.num_h, dtype=dtype),
                np.linspace(0, 2*np.pi, self.num_w, dtype=dtype)]
        self.X, self.Y = np.meshgrid(x, y)
        self.grid = np.stack([self.X, self.Y], axis=0)
        self.degen_type = degen_type
        self.scale = scale
        self.num_frames = num_frames
        # the following is for Kolmogorov flow
        u0 = 1.0
        rey = np.linspace(100, 1050, 20, endpoint=True)
        sigma = np.linspace(2, 8, 7, endpoint=True)
        r, s = np.meshgrid(rey, sigma, indexing='ij')
        self.rey = r.reshape(-1)
        self.num_rey = len(self.rey)

        self.vis = u0 / self.rey
        self.f = [1. * np.sin(k * self.y) for k in s.reshape(-1)]

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames]
        x = rearrange(x, 't h w c -> (t c) h w')
        shift_x, shift_y = np.random.choice(self.num_h, 1)[0], np.random.choice(self.num_w, 1)[0]   # data aug.
        x = np.roll(x, (shift_x, shift_y), axis=(1, 2))     # data aug.
        if 'fi' in self.degen_type:
            x_lr = x[:, ::self.scale, ::self.scale]
        elif 'spec_real' in self.degen_type:
            x_lr = np.real(np.fft.fft(x[:, ::self.scale, ::self.scale], axis=(-2, -1)))
        if self.is_condition:
            r = self.vis[i_b%self.num_rey]*100*np.ones_like(self.y)[np.newaxis]
            f = self.f[i_b%self.num_rey][np.newaxis]
            return x_lr, x, np.concatenate([f, r], axis=0), self.grid      # B T C H W
        else:
            return x_lr, x, self.grid       # B T C H W

    def __len__(self):
        return self.length
