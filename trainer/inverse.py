import torch
import torch.nn.functional as F


def get_interp_fn(config):
    if 'fi' in config.degen_type:
        interp_fn = lambda x: F.interpolate(x, scale_factor=config.scale, mode=config.interp_method, align_corners=False)
    elif 'spec_real' in config.degen_type:
        interp_fn = lambda x: F.pad(x, pad=(int(config.image_size*(1-1./config.scale))//2, int(config.image_size*(1-1./config.scale))//2), mode='constant', value=0)
    return interp_fn
