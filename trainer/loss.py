import torch
import einops
import torch.nn.functional as F
import torch.nn as nn


class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()


def loss_fn_video(net, sde, batch, eps=1e-5, **kwarg):
    # batch: B T C H W
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    std_e = std[:, None, None, None, None] if len(batch.shape)==5 else std[:, None, None, None]
    perturbed_data = mean + std_e * z 

    kwarg['x0'] = batch
    kwarg['timesteps'] = std
    score, _ = net(perturbed_data, **kwarg)
    losses = torch.square(score * std_e + z)
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss


def sample_noise(shape, channel_modal, device='cpu', dtype=torch.float32):
    # shape: [b, c, h, w], stype: list, 0 for separate sampling, 1 for integrated sampling
    z = torch.randn(shape[0], shape[-1] * shape[-2] * (shape[-3] - 1) + 1).to(device).type(dtype)
    z = torch.cat([z[:, :-1].reshape(shape[0], shape[1]-1, *shape[2:]),
                   z[:, -1:, None, None] * torch.ones(shape[0], 1, *shape[2:],
                                                                    device=device, dtype=dtype)],
                  dim=1)
    return z


loss_l1 = MatchingLoss(loss_type='l1')
def loss_fn_inverse(net, batch_lr, batch, cond=None, grid=None, cat_dim=1):
    if cond is not None:
        inp = torch.cat([batch_lr, cond], dim=cat_dim)
    else:
        inp = batch_lr
    if grid is not None:
        pred = net(inp, grid)
    else:
        pred = net(inp)
    return loss_l1(pred.squeeze(), batch.squeeze())



def predict_fn(net, sde, x, t, continuous=True):
    if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
    else:
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()
    score = net(x, labels)
    return score


def voriticity_residual(w, num_frame=5, dt=0.1, scalar=None):
    # w [b t h w]
    device = w.device
    batchsize = w.size(0)
    # w = w.clone()
    if scalar is not None:
        scalar_std = torch.ones([1, len(w[0]), 1, 1]).to(device)
        scalar_mean = torch.zeros([1, len(w[0]), 1, 1]).to(device)
        scalar_std[:, :5] = scalar_std[:, :5]*scalar.std
        scalar_mean[:, :5] = scalar_mean[:, :5]+scalar.mean
        w = w*scalar_std+scalar_mean
    # w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)

    w_h = torch.fft.fft2(w[:, 1:num_frame-1], dim=[2, 3])
    re = torch.mean(w[:, -1].view(len(w), -1), dim=1)[:, None, None, None]
    if scalar is not None:
        re = 1000*re
    f_h = torch.fft.fft2(w[:, num_frame:num_frame+1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    ks = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0)
    k_x, k_y = torch.meshgrid(ks, ks, indexing='ij')
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h
    fy_h = 1j * k_y * f_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    f = -torch.fft.irfft2(fy_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy

    wt = (w[:, 2:num_frame, :, :] - w[:, :num_frame-2, :, :]) / (2 * dt)

    # establish forcing term
    # x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    # x = x[0:-1]
    # X, Y = torch.meshgrid(x, x)
    # f = -4*torch.cos(4*Y)

    residual = wt + (advection - (1.0 / re) * wlap + 0.1*w[:, 1:num_frame-1]) - f
    residual_loss = (residual**2).mean()
    # dw = torch.autograd.grad(residual_loss, w)[0]
    return residual_loss, torch.sum((w[:, :num_frame]**2).reshape(len(w), -1), dim=1)[:, None, None, None]


def kse_residual(w, num_frame=10, dt=0.5, scalar=None):
    # w [b t h w]
    device = w.device
    batchsize = w.size(0)
    # w = w.clone()
    vis = w[:, :1, 1].detach().mean()*4.+3.
    u = w[:, :, 0]      # b t h
    if scalar is not None:
        u = scalar(u)
    # u.requires_grad_(True)
    nx = u.size(2)

    u_h = torch.fft.fft(u[:, 1:num_frame-1], dim=2)
    u2_h = torch.fft.fft(u[:, 1:num_frame-1]**2, dim=2)
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    # k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
    #                 torch.arange(start=-k_max, end=0, step=1, device=device)), 0)
    k = (torch.conj(torch.cat((torch.arange(0, N/2), torch.tensor([0]), torch.arange(-N/2+1, 0)))) / 16).to(device)
    # Negative Laplacian in Fourier space
    uux_h = 1j * k * u2_h * 0.5
    uxx_h = (1j * k)**2 * u_h
    u4x_h = (1j * k)**4 * u_h

    uux = torch.fft.irfft(uux_h[..., :k_max + 1], dim=-1)
    uxx = torch.fft.irfft(uxx_h[..., :k_max + 1], dim=-1)
    u4x = torch.fft.irfft(u4x_h[..., :k_max + 1], dim=-1)

    ut = (u[:, 2:num_frame] - u[:, :num_frame-2]) / (2 * dt)

    # establish forcing term
    # x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    # x = x[0:-1]
    # X, Y = torch.meshgrid(x, x)
    # f = -4*torch.cos(4*Y)

    residual = ut + uux + uxx + vis*u4x
    residual_loss = (residual**2).mean()
    # dw = torch.autograd.grad(residual_loss, w)[0]
    return residual_loss, residual
