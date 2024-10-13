# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .unet_video import UNetVideoModel
from .ema import ExponentialMovingAverage
from .fno import FNO1d, FNO2d, FNO3d
from .deeponet import DeepONet1D, DeepONet2D, MIONet2D
from .lno import LNO1d, LNO2d
from .unet import UNet1d, UNet2d, UNet3d
import numpy as np
import torch


def get_model(config):
    kwargs = dict(in_channels=config.num_channels, 
                  model_channels=config.nf, 
                  out_channels=config.num_channels, 
                  num_res_blocks=config.num_res_blocks, 
                  attention_resolutions=config.attn_resolutions, 
                  image_size=config.image_size, 
                  dropout=config.dropout, 
                  channel_mult=config.ch_mult,
                  conv_resample=True,
                  dims=config.dims,
                  num_heads=config.num_heads,
                  use_rpe_net=True)
    return UNetVideoModel(**kwargs)


def get_model_inverse(config):
    initial_step = 1
    in_channels = config.num_components*config.num_frames+config.num_conditions
    out_channels = config.num_components*config.num_frames
    if config.dims == 1:
        if 'fno' in config.model_name.lower():
            model = FNO1d(num_channels=in_channels,
                          out_channels=out_channels,
                            width=config.width,
                            modes=config.modes,
                            initial_step=initial_step)
        elif 'deeponet' in config.model_name.lower():
            model = DeepONet1D(in_channels=in_channels,
                                out_dims=out_channels,
                                init_channels=config.width,
                                mid_dim=config.mid_dim_deeponets)
        elif 'lno' in config.model_name.lower():
            x = np.linspace(0, 1, config.image_size, dtype='float32')
            model = LNO1d(in_channels=in_channels,
                        out_channels=out_channels,
                        width=config.width,
                        modes=config.modes,
                        tx=torch.from_numpy(x))
        elif 'unet' in config.model_name.lower():
            model = UNet1d(in_channels, out_channels, init_features=config.width)
        else:
            raise NotImplementedError('Not implemented model!')
    elif config.dims == 2:
        if 'fno' in config.model_name:
            model = FNO2d(num_channels=in_channels,
                          out_channels=out_channels,
                            width=config.width,
                            modes1=config.modes,
                            modes2=config.modes,
                            initial_step=initial_step)
        elif 'deeponet' in config.model_name:
            model = DeepONet2D(in_channels=in_channels,
                                out_dims=out_channels,
                                init_channels=config.width,
                                mid_dim=config.mid_dim_deeponets)
        elif 'mionet' in config.model_name.lower():
            model = MIONet2D(in_channels=in_channels,
                                out_dims=out_channels,
                                init_channels=config.width,
                                mid_dim=config.mid_dim_deeponets)
        elif 'lno' in config.model_name:
            x, y = [np.linspace(0, 2*np.pi, config.image_size, dtype='float32'),
                            np.linspace(0, 2*np.pi, config.image_size, dtype='float32')]
            model = LNO2d(in_channels=in_channels,
                        out_channels=out_channels,
                        width=config.width,
                        modes1=config.modes, modes2=config.modes,
                        tx=torch.from_numpy(x[np.newaxis]), ty=torch.from_numpy(y[np.newaxis]))
        elif 'unet' in config.model_name.lower():
            model = UNet2d(in_channels, out_channels, init_features=config.width)
        else:
            raise NotImplementedError('Not implemented model!')
    return model


def get_ema(parameters, decay, use_num_updates=True):
    return ExponentialMovingAverage(parameters, decay, use_num_updates=use_num_updates)
