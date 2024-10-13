import torch
import numpy as np
import abc
import functools
from tqdm import tqdm
from scipy import integrate
from sampler.sde import VESDE, VPSDE
from trainer.loss import predict_fn, voriticity_residual, sample_noise, kse_residual
from einops import rearrange
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import Voronoi, cKDTree


class ODE_Solver(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, net_fn, eps=None):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        if sde.config.sde != 'poisson':
            self.rsde = sde.reverse(net_fn, probability_flow=True)
        self.net_fn = net_fn
        self.eps = eps

    @abc.abstractmethod
    def update_fn(self, x, t, t_list=None, idx=None):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, net_fn, probability_flow=False, channel_modal=None, eps=None):
        super().__init__()
        self.sde = sde
        self.channel_modal = channel_modal
        # Compute the reverse SDE/ODE
        if sde.config.sde != 'poisson':
            self.rsde = sde.reverse(net_fn, probability_flow)
        self.net_fn = net_fn
        self.eps = eps

    @abc.abstractmethod
    def update_fn(self, x, t, t_list=None, idx=None):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
        super().__init__()
        self.sde = sde
        self.net_fn = net_fn
        self.snr = snr
        self.n_steps = n_steps
        self.channel_modal = channel_modal

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, net_fn, probability_flow=False, eps=None):
        super().__init__(sde, net_fn, probability_flow, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        z = torch.randn_like(x)
        if self.sde.config.sde == 'poisson':
            if t_list is None:
                dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt = - (1 - torch.exp(t_list[idx + 1] - t_list[idx]))
                dt = float(dt.cpu().numpy())
            drift = self.sde.ode(self.net_fn, x, t)
            diffusion = torch.zeros((len(x)), device=x.device)
        else:
            if t_list is None:
                dt = -1. / self.sde.N
            drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


class ForwardEulerPredictor(ODE_Solver):
    def __init__(self, sde, net_fn, eps=None):
        super().__init__(sde, net_fn, eps)

    def update_fn(self, x, t, t_list=None, idx=None):

        if self.sde.config.sde == 'poisson':
            # dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            drift = self.sde.ode(self.net_fn, x, t)
            if t_list is None:
                dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt = - (1 - torch.exp(t_list[idx + 1] - t_list[idx]))
                dt = float(dt.cpu().numpy())
        else:
            dt = -1. / self.sde.N
            drift, _ = self.rsde.sde(x, t)
        x = x + drift * dt
        return x


class ImprovedEulerPredictor(ODE_Solver):
    def __init__(self, sde, net_fn, eps=None):
        super().__init__(sde, net_fn, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        if self.sde.config.sde == 'poisson':
            if t_list is None:
                dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt = (torch.exp(t_list[idx + 1] - t_list[idx]) - 1)
                dt = float(dt.cpu().numpy())
            drift = self.sde.ode(self.net_fn, x, t)
        else:
            dt = -1. / self.sde.N
            drift, _ = self.rsde.sde(x, t)
        x_new = x + drift * dt

        if idx == self.sde.N - 1:
            return x_new
        else:
            idx_new = idx + 1
            t_new = t_list[idx_new]
            t_new = torch.ones(len(t), device=t.device) * t_new

            if self.sde.config.sde == 'poisson':
                if t_list is None:
                    dt_new = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
                else:
                    # integration over z
                    dt_new = (1 - torch.exp(t_list[idx] - t_list[idx + 1]))
                    dt_new = float(dt_new.cpu().numpy())
                drift_new = self.sde.ode(self.net_fn, x_new, t_new)
            else:
                drift_new, diffusion = self.rsde.sde(x_new, t_new)
                dt_new = -1. / self.sde.N

            x = x + (0.5 * drift * dt + 0.5 * drift_new * dt_new)
            return x


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, net_fn, probability_flow=False, channel_modal=None, eps=None):
        super().__init__(sde, net_fn, probability_flow, channel_modal, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        t_shape = t.shape
        f, G = self.rsde.discretize(x, t, self.channel_modal)
        z = torch.randn_like(x)
        x_mean = x - f
        G = G.view(*t_shape)
        if self.channel_modal is None:
            x = x_mean + G[:, None, None, None] * z
        else:
            G = G.repeat_interleave(torch.tensor(self.channel_modal).to(G.device), dim=1)
            x = x_mean + G[:, :, None, None] * z
        return x, x_mean


class ReverseDiffusionPredictorMM(Predictor):
    def __init__(self, sde, net_fn, probability_flow=False, channel_modal=None, eps=None):
        super().__init__(sde, net_fn, probability_flow, channel_modal, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        t_shape = t.shape
        f, G = self.rsde.discretize(x, t, self.channel_modal)
        z = sample_noise(x.shape, channel_modal=self.channel_modal, device=x.device, dtype=x.dtype)
        x_mean = x - f
        G = G.view(*t_shape)
        if self.channel_modal is None:
            x = x_mean + G[:, None, None, None] * z
        else:
            G = G.repeat_interleave(torch.tensor(self.channel_modal).to(G.device), dim=1)
            x = x_mean + G[:, :, None, None] * z
        return x, x_mean


class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, net_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, t_list=None, idx=None):
        return x, x



class LangevinCorrector(Corrector):
  def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
    super().__init__(sde, net_fn, snr, n_steps, channel_modal)
    if not isinstance(sde, VPSDE) \
        and not isinstance(sde, VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    net_fn = self.net_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, VPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones(len(x)).float().to(t.device)

    for i in range(n_steps):
      grad = net_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      if len(x.shape) > 4:
        x_mean = x + step_size[:, None, None, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None, None] * noise
      else:
        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


class LangevinCorrectorMM(Corrector):
    def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
        super().__init__(sde, net_fn, snr, n_steps, channel_modal)
        if not isinstance(sde, VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        net_fn = self.net_fn
        n_steps = self.n_steps
        target_snr = self.snr

        if isinstance(sde, VESDE):
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = net_fn(x, t)
            noise = sample_noise(x.shape, channel_modal=self.channel_modal, device=x.device, dtype=x.dtype)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            if self.channel_modal is None:
                x_mean = x + step_size[:, None, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise
            else:
                step_size = step_size.repeat_interleave(torch.tensor(self.channel_modal).to(step_size.device), dim=1)
                x_mean = x + step_size[:, :, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, :, None, None] * noise

        return x, x_mean


class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, net_fn, snr, n_steps):
        super().__init__(sde, net_fn, snr, n_steps)
        if not isinstance(sde, VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        net_fn = self.net_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VESDE):
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = net_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
        pass

    def update_fn(self, x, t):
        return x, x


def shared_ode_solver_update_fn(x, t, sde, net, ode_solver, eps, t_list=None, idx=None):
    """A wrapper that configures and returns the update function of ODE solvers."""
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    ode_solver_obj = ode_solver(sde, net_fn, eps)
    return ode_solver_obj.update_fn(x, t, t_list=t_list, idx=idx)


def shared_predictor_update_fn(x, t, sde, net, predictor, probability_flow, continuous, eps,
                               channel_modal=None, t_list=None, idx=None):
    """A wrapper that configures and returns the update function of predictors."""
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, net_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, net_fn, probability_flow, channel_modal, eps)
    return predictor_obj.update_fn(x, t, t_list=t_list, idx=idx)


def shared_corrector_update_fn(x, t, sde, net, corrector, continuous, snr, n_steps, channel_modal=None):
    """A wrapper tha configures and returns the update function of correctors."""
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, net_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, net_fn, snr, n_steps, channel_modal=channel_modal)
    return corrector_obj.update_fn(x, t)


def ode_sampler(net, sde, ode_solver, shape, device='cpu', dtype='float32', eps=1e-3):
    ode_update_fn = functools.partial(shared_ode_solver_update_fn,
                                      sde=sde,
                                      ode_solver=ode_solver,
                                      eps=eps)
    x = sde.prior_sampling(shape).to(device).float()
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()

    xs = []
    for i in tqdm(range(sde.N), desc='generating...', total=sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device).float() * t
        x = ode_update_fn(x, vec_t, net=net, t_list=timesteps, idx=i)
        xs.append(x)
    return x, sde.N


def pc_sampler(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3):
    dtype_torch = getattr(torch, dtype)
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous,
                                            eps=eps,
                                            channel_modal=config.channel_modal)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            channel_modal=config.channel_modal)
    # if is_mm:
    #     x = sample_noise(shape, channel_modal=config.channel_modal, device=device, dtype=dtype_torch)*sde.sigma_max
    # else:
    x = sde.prior_sampling(shape).to(device).float()
    x0 = torch.tensor(x0, device=device).float()
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    # if 'mm' in config.version:
    #     mode = np.array(config.mm_mode)
    #     mode_r = torch.tensor(mode.repeat(config.channel_modal, 0), device=device).bool()

    x_generated = [x.detach().cpu().numpy()]
    for i in tqdm(range(sde.N)):
        t = timesteps[i]
        # if 'mm' in config.version:
        #     t = torch.tensor(util.mode_to_ts(mode, pos=eps_t, neg=t), device=device).float()
        #     vec_t = torch.ones([shape[0], config.num_modals], device=t.device).float() * t[None, :]
        #     x = x * (~mode_r)[None, :, None, None] + x0 * mode_r[None, :, None, None]
        #     if 'cond' in config.version:
        #         vec_t = torch.cat([vec_t, torch.ones_like(vec_t[:, :1])*pattern], dim=1)
        # else:
        vec_t = torch.ones(shape[0], device=t.device).float() * t
        x, x_mean = corrector_update_fn(x, vec_t, net=net)
        # if 'mm' in config.version:
        #     x = x * (~mode_r)[None, :, None, None] + x0 * mode_r[None, :, None, None]
        x, x_mean = predictor_update_fn(x, vec_t, net=net)
        x_generated.append(x_mean.detach().cpu().numpy())

    return x_mean if denoise else x, x_generated


def pc_sampler_video_ar(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False, alpha=1., mult=10, transform_init=None, num_steps=10, overlap=1,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            )
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    x0 = torch.tensor(x0, device=device).float()            # batch*ol*(c+npara)*h*w
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()

    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(ns//(nf-ol)+1)      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [len(x0), nf, ncomp+config.num_conditions, config.image_size, config.image_size]       # batch*nf*(c+npara)*h*w

    transform_ar, alpha_ar = lambda x: x[:, :ol], alpha
    transform_init, alpha_init = transform_ar if transform_init is None else transform_init, alpha*mult

    x_generated = []
    pred = []
    for i_b in range(b):
        if i_b == 0:
            y = x0
            transform = transform_init
            alpha = alpha_init
        else:
            y = x_mean[:, -ol:].detach()
            transform = transform_ar
            alpha = alpha_ar
        x = sde.prior_sampling(shape).to(device).float()
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device).float() * t
            '''corrector'''
            x, x_mean = corrector_update_fn(x, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            with torch.enable_grad():
                inp = x.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)
                with torch.no_grad():
                    f, G = sde.discretize(x, vec_t)
                    rev_f = f - G[:, None, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    x_mean = x - rev_f
                    x_u = x_mean + rev_G[:, None, None, None, None] * z
                    # x_u = x
                # dps loss
                _, std = sde.marginal_prob(x, t)
                x0_hat = std**2*score + inp
                loss_dps = ((y-transform(x0_hat))**2).sum()
                loss = alpha * loss_dps
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                x = x_u - dx       # /torch.sqrt(scalar1.mean())
                # x = x_u
            x = x.detach()
        pred.append(x_mean if i_b==0 else x_mean[:, ol:])
    return torch.cat(pred, dim=1).detach().cpu().numpy() if denoise else x, x_generated


def pc_sampler_video1d_ar(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False, alpha=1., mult=10, transform_init=None, num_steps=10, overlap=1,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,)
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    x0 = torch.tensor(x0, device=device).float()            # batch*ol*(c+npara)*h*w
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()

    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(ns//(nf-ol)+1)      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [config.num_samples, nf, ncomp+config.num_conditions, config.image_size]       # batch*nf*(c+npara)*h

    transform_ar, alpha_ar = lambda x: x[:, :ol], alpha
    transform_init, alpha_init = transform_ar if transform_init is None else transform_init, alpha*mult

    x_generated = []
    pred = []
    for i_b in range(b):
        if i_b == 0:
            y = x0
            transform = transform_init
            alpha = alpha_init
        else:
            y = x_mean[:, -ol:].detach()
            transform = transform_ar
            alpha = alpha_ar
        x = sde.prior_sampling(shape).to(device).float()
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device).float() * t
            '''corrector'''
            x, x_mean = corrector_update_fn(x, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            with torch.enable_grad():
                inp = x.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)
                with torch.no_grad():
                    f, G = sde.discretize(x, vec_t)
                    rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    x_mean = x - rev_f
                    x_u = x_mean + rev_G[:, None, None, None] * z
                    # x_u = x
                # dps loss
                _, std = sde.marginal_prob(x, t)
                x0_hat = std**2*score + inp
                loss_dps = ((y-transform(x0_hat))**2).sum()
                loss = alpha * loss_dps
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                x = x_u - dx       # /torch.sqrt(scalar1.mean())
                # x = x_u
            x = x.detach()
        pred.append(x_mean if i_b==0 else x_mean[:, ol:])
    return torch.cat(pred, dim=1).detach().cpu().numpy() if denoise else x, x_generated


def s3gm_sample_2d(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                T_prime_y=10, T_prime=0, overlap=1,
                              device='cpu', dtype='float32', eps=1e-12, 
                              probability_flow=False, continuous=True):
    if T_prime_y > 0:
        x_y, _, _ = complete_video_pc_dps(config, net, sde, y, transform, corrector, n_steps=n_steps, 
                                    alpha=alpha, beta=0, gamma1=beta, gamma2=beta, num_steps=T_prime_y, overlap=overlap,
                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False, 
                                    probability_flow=probability_flow, continuous=continuous)
    else:
        x_y = y
    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1] - 1
        transform_init = lambda x: x[:, :overlap]
        x_extra, _ = pc_sampler_video_ar(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0, 
                                    n_steps=n_steps, probability_flow=probability_flow, 
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
    return np.concatenate([x_y, x_extra[:, overlap:]], axis=1) if T_prime > T_prime_y else x_y


def s3gm_sample_1d(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                T_prime_y=10, T_prime=0, overlap=1,
                              device='cpu', dtype='float32', eps=1e-12, 
                              probability_flow=False, continuous=True):
    if T_prime_y > 0:
        x_y, _, _ = complete_video1d_pc_dps(config, net, sde, y, transform, corrector, n_steps=n_steps, 
                                    alpha=alpha, beta=0, gamma1=beta, gamma2=beta, num_steps=T_prime_y, overlap=overlap,
                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False, 
                                    probability_flow=probability_flow, continuous=continuous)
    else:
        x_y = y
    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1] - 1
        transform_init = lambda x: x[:, :overlap]
        x_extra, _ = pc_sampler_video1d_ar(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0, 
                                    n_steps=n_steps, probability_flow=probability_flow, 
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
    return np.concatenate([x_y, x_extra[:, overlap:]], axis=1) if T_prime > T_prime_y else x_y


def complete_video_pc_dps(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=None, gamma1=100., gamma2=100, snr=0.128, std_y=None, gamma=1.e-2,
                          num_steps=10, overlap=1,
                              device='cpu', dtype='float32', eps=1e-3, save_sample_path=False,
                              probability_flow=False, continuous=True, data_scalar=None):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,)
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    # x_known = torch.from_numpy(x0).to(device).type(dtype_torch)
    y = torch.from_numpy(y).to(device).type(dtype_torch)
    # shape_sample = [len(y), config.num_channels, config.image_size, config.image_size]

    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(ns//(nf-ol)+1)      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    nc = config.num_conditions
    shape = [len(y), b, nf, ncomp+nc, config.image_size, config.image_size]       # batch*b*nf*(c+npara)*h*w
    # shape_sample = [config.num_samples, ns_real, ncomp+nc, config.image_size, config.image_size]     # batch*ns_real*(c+npara)*h*w

    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([config.num_samples, ns_real, ncomp+nc, config.image_size, config.image_size], dtype=dtype_torch, device=device)   # batch*ns_real*(c+npara)*h*w
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        # sample[:, :, ncomp:] = xx[:, 0, 0:1, ncomp:]
        return sample

    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    x_unknown = sde.prior_sampling(shape).to(device).float()    # batch*b*nf*(c+npara)*h*w
    # x_u_temp = sde.prior_sampling(shape_sample)      # batch*ns_real*(c+npara)*h*w
    # x_unknown = []
    # for i in range(b):
    #     x_unknown.append(x_u_temp[:, i*(nf-ol):i*(nf-ol)+nf])
    # x_unknown = torch.stack(x_unknown, dim=1).float().to(device)
    
    x = x_unknown           # batch*b*(nf*c+npara)*h*w
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[])
    with tqdm(range(sde.N)) as tqdm_setting:
        for i in range(sde.N):
            t = timesteps[i]

            '''method 1 (batched)'''
            vec_t = torch.ones(shape[0]*b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')       # (batch*b)*nf*(c+npara)*h*w

            '''corrector'''
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)     # (batch*b)*nf*(c+npara)*h*w

            '''predictor'''
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h w -> (b n) t c h w')

            with torch.enable_grad():
                inp = temp.clone()                  # (batch*b)*nf*(c+npara)*h*w
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)          # (batch*b)*nf*(c+npara)*h*w
                with torch.no_grad():
                    f, G = sde.discretize(temp, vec_t)
                    rev_f = f - G[:, None, None, None, None] ** 2 * score.detach() * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    temp_mean = temp - rev_f
                    temp_u = temp_mean + rev_G[:, None, None, None, None] * zb
                # dps loss
                _, std = sde.marginal_prob(xb, vec_t)
                if isinstance(sde, VESDE):
                    x0_hat = rearrange(std[:, None, None, None, None] ** 2 * score + inp, '(b n) t c h w -> b n t c h w', n=b)     # batch*b*nf*(c+npara)*h*w
                else:
                    alpha_sqrt_ = (1-std**2).sqrt()[:, None, None, None, None]
                    x0_hat = rearrange((std[:, None, None, None, None] ** 2 * score + inp)/alpha_sqrt_, '(b n) t c h w -> b n t c h w', n=b)     # batch*b*nf*(c+npara)*h*w
                x0_hat_temp = x_to_sample(x0_hat)
                if save_sample_path:
                    x0_hats.append(x0_hat.detach().cpu().numpy())

                var = std_y**2 + gamma * std**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_temp)) ** 2 / var).reshape(x0_hat.shape[0], -1), dim=-1)  # /scalar.sqrt()  *std[None, :, None, None]
                loss_dps = torch.sum(loss_dps, dim=0)  # /loss_dps.detach().mean().sqrt()
                if std_y is not None:
                    loss_dps = loss_dps/2.
                # loss_dps = loss_dps/loss_dps.detach().sqrt()    # normalize

                loss_consis = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, :ncomp].detach()-x0_hat[:, 1:, :ol, :ncomp])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)    # , len(x0_hat[0])
                loss_consis = torch.sum(loss_consis, dim=-1).mean()        # *torch.softmax(loss_consis.detach(), dim=1)
                # loss_consis = loss_consis/loss_consis.detach().sqrt()    # normalize
                # loss_consis_para = torch.sum(((x0_hat[:, 1:, :, ncomp:]-x0_hat[:, 0:1, :, ncomp:].detach())**2).reshape(x0_hat.shape[0], -1), dim=-1)
                loss_consis_para = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, ncomp:].detach()-x0_hat[:, 1:, :ol, ncomp:])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()
                # loss_consis_para = loss_consis_para/loss_consis_para.detach().sqrt()    # normalize

                if config.physics_guide:
                    loss_eq, scalar2 = voriticity_residual(x0_hat, ns_real, 1., data_scalar)
                    scalar2 = scalar2.detach()
                    loss = alpha * loss_dps + beta * loss_eq + gamma1 * loss_consis + gamma2 * loss_consis_para  # /loss_dps.detach().sqrt()  /scalar2.mean().sqrt()
                    tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss dps: {alpha * loss_dps.item():.5e} | loss eq: {beta(t) * loss_eq.item():.5e} | loss consis: {gamma1 * loss_consis.item():.5e}')
                    losses['loss'].append(loss.item())
                    losses['loss_eq'].append(loss_eq.item())
                    losses['loss_dps'].append(loss_dps.item())
                    losses['loss_consis'].append(loss_consis.item())
                    losses['loss_consis_para'].append(loss_consis_para.item())
                    assert (not torch.isnan(loss_eq))
                else:
                    loss = alpha * loss_dps + gamma1 * loss_consis + gamma2 * loss_consis_para
                    tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss dps: {alpha * loss_dps.item():.5e} | loss consis: {gamma1 * loss_consis.item():.5e}')
                    losses['loss'].append(loss.item())
                    losses['loss_dps'].append(loss_dps.item())
                    losses['loss_consis'].append(loss_consis.item())
                    losses['loss_consis_para'].append(loss_consis_para.item())
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                temp = temp_u - dx     # (batch*b)*(nf*c+npara)*h*w
            #     # x = x_u
            temp = temp.detach()

            x = rearrange(temp, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(temp_mean, '(b n) t c h w -> b n t c h w', n=b)
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            tqdm_setting.update(1)

    return x_to_sample(x_mean).detach().cpu().numpy(), x_generated if save_sample_path else None, losses


def complete_video1d_pc_dps(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=None, gamma1=100., gamma2=100, snr=0.128, std_y=None, gamma=1.e-2,
                          num_steps=10, overlap=1,
                              device='cpu', dtype='float32', eps=1e-3, save_sample_path=False,
                              probability_flow=False, continuous=True, data_scalar=None):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,)
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    # x_known = torch.from_numpy(x0).to(device).type(dtype_torch)
    y = torch.from_numpy(y).to(device).type(dtype_torch)
    # shape_sample = [len(y), config.num_channels, config.image_size, config.image_size]

    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(ns//(nf-ol)+1)      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [len(y), b, nf, ncomp+config.num_conditions, config.image_size]       # batch*b*nf*(c+npara)*h
    # shape_sample = [config.num_samples, ns_real, ncomp+config.num_modals-1, config.image_size]     # batch*ns_real*(c+npara)*h

    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp+config.num_conditions, config.image_size], dtype=dtype_torch, device=device)   # batch*ns_real*(c+npara)*h
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    x_unknown = sde.prior_sampling(shape).to(device).float()    # batch*b*nf*(c+npara)*h
    
    x = x_unknown           # batch*b*(nf*c+npara)*h*w
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[])
    with tqdm(range(sde.N)) as tqdm_setting:
        for i in range(sde.N):
            t = timesteps[i]

            '''method 1 (batched)'''
            vec_t = torch.ones(shape[0]*b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h -> (b n) t c h')       # (batch*b)*nf*(c+npara)*h*w

            '''corrector'''
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)     # (batch*b)*nf*(c+npara)*h*w

            '''predictor'''
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h -> (b n) t c h')

            with torch.enable_grad():
                inp = temp.clone()                  # (batch*b)*nf*(c+npara)*h*w
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)          # (batch*b)*nf*(c+npara)*h*w
                with torch.no_grad():
                    f, G = sde.discretize(temp, vec_t)
                    rev_f = f - G[:, None, None, None] ** 2 * score.detach() * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    temp_mean = temp - rev_f
                    temp_u = temp_mean + rev_G[:, None, None, None] * zb
                # dps loss
                _, std = sde.marginal_prob(xb, vec_t)
                if isinstance(sde, VESDE):
                    x0_hat = rearrange(std[:, None, None, None] ** 2 * score + inp, '(b n) t c h -> b n t c h', n=b)     # batch*b*nf*(c+npara)*h
                else:
                    alpha_sqrt_ = (1-std**2).sqrt()[:, None, None, None]
                    x0_hat = rearrange((std[:, None, None, None] ** 2 * score + inp)/alpha_sqrt_, '(b n) t c h -> b n t c h', n=b)     # batch*b*nf*(c+npara)*h
                x0_hat_temp = x_to_sample(x0_hat)
                if save_sample_path:
                    x0_hats.append(x0_hat.detach().cpu().numpy())

                var = std_y**2 + gamma * std**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_temp)) ** 2 / var).reshape(x0_hat.shape[0], -1), dim=-1)  # /scalar.sqrt()  *std[None, :, None, None]
                loss_dps = torch.sum(loss_dps, dim=0)  # /loss_dps.detach().mean().sqrt()
                if std_y is not None:
                    loss_dps = loss_dps/2.
                # loss_dps = loss_dps/loss_dps.detach().sqrt()    # normalize

                loss_consis = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, :ncomp].detach()-x0_hat[:, 1:, :ol, :ncomp])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)    # , len(x0_hat[0])    /std[:, None, None].sqrt()
                loss_consis = torch.sum(loss_consis)        # *torch.softmax(loss_consis.detach(), dim=1)
                # loss_consis = loss_consis/loss_consis.detach().sqrt()    # normalize
                loss_consis_para = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, ncomp:]-x0_hat[:, 1:, :ol, ncomp:].detach())**2).reshape(x0_hat.shape[0], -1), dim=-1)
                loss_consis_para = torch.sum(loss_consis_para, dim=0)
                # loss_consis_para = loss_consis_para/loss_consis_para.detach().sqrt()    # normalize

                if config.physics_guide:
                    loss_eq, _ = kse_residual(inp, nf, 0.5, data_scalar)       # x0_hat_temp, ns_real
                    # scalar2 = scalar2.detach()
                    loss_eq = loss_eq/loss_eq.detach().sqrt()
                    loss = alpha * loss_dps + beta(t) * loss_eq + gamma1 * loss_consis + gamma2 * loss_consis_para  # /loss_dps.detach().sqrt()  /scalar2.mean().sqrt()
                    tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss dps: {alpha * loss_dps.item():.5e} | loss eq: {beta(t) * loss_eq.item():.5e} | loss consis: {gamma1 * loss_consis.item():.5e}')
                    losses['loss'].append(loss.item())
                    losses['loss_eq'].append(loss_eq.item())
                    losses['loss_dps'].append(loss_dps.item())
                    losses['loss_consis'].append(loss_consis.item())
                    losses['loss_consis_para'].append(loss_consis_para.item())
                    assert (not torch.isnan(loss_eq))
                else:
                    loss = alpha * loss_dps + gamma1 * loss_consis + gamma2 * loss_consis_para
                    tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss dps: {alpha * loss_dps.item():.5e} | loss consis: {gamma1 * loss_consis.item():.5e}')
                    losses['loss'].append(loss.item())
                    losses['loss_dps'].append(loss_dps.item())
                    losses['loss_consis'].append(loss_consis.item())
                    losses['loss_consis_para'].append(loss_consis_para.item())
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                temp = temp_u - dx     # (batch*b)*(nf*c+npara)*h*w
            #     # x = x_u
            temp = temp.detach()

            x = rearrange(temp, '(b n) t c h -> b n t c h', n=b)
            x_mean = rearrange(temp_mean, '(b n) t c h -> b n t c h', n=b)
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            tqdm_setting.update(1)

    return x_to_sample(x_mean).detach().cpu().numpy(), x0_hats if save_sample_path else None, losses


def vor_cal(u, v, grid_num, x_range):
    dx = (x_range[1]-x_range[0])/grid_num
    vor = (v[:-1, 1:]-v[:-1, :-1])/dx-(u[1:, :-1]-u[:-1, :-1])/dx
    return vor


def vor_cal_batch(x, grid_num, x_range, reverse=False, method='diff_1st', is_stagger=True):
    # method: 'diff_1st', 'spectral'
    vor = []
    for v in x:
        vx, vy = (v[1], v[0]) if reverse else (v[0], v[1])
        if 'diff_1st' in method:
            vor.append(vor_cal(vx, vy, grid_num, x_range))
        elif 'spectral' in method:
            vor.append(vor_cal_spectral(vx, vy, is_stagger=is_stagger))
        else:
            raise NotImplementedError('No such method for vorticity calculation!')
    return np.array(vor)


def vor_cal_plus(u, v, grid_num, x_range):
    omega = np.zeros((grid_num, grid_num))
    dx = dy = (x_range[1] - x_range[0]) / grid_num
    for i in range(1, grid_num - 1):
        for j in range(1, grid_num - 1):
            dudy = (u[i, j + 1] - u[i, j - 1]) / (2 * dy)
            dvdx = (v[i + 1, j] - v[i - 1, j]) / (2 * dx)
            omega[i, j] = dvdx - dudy
    return omega


def vor_cal_spectral(u, v, is_stagger=True):
    if is_stagger:
        # for staggered grid arrangement, we interpolate velocities from cell faces to cell centres
        u = 0.5 * (u + np.roll(u, 1, axis=1))
        v = 0.5 * (v + np.roll(v, -1, axis=0))
    k_max = len(u)//2
    k = np.concatenate([np.arange(0, k_max, 1), np.arange(-k_max, 0, 1)])
    k_x, k_y = np.meshgrid(k, k)
    F_u = np.fft.fft2(u)
    F_v = np.fft.fft2(v)
    # F_ux = 1j * k_x * F_u
    F_uy = 1j * k_y * F_u
    F_vx = 1j * k_x * F_v
    # F_vy = 1j * k_y * F_v
    # ux = np.fft.ifft2(F_ux)
    uy = np.fft.irfft2(F_uy[..., :k_max+1])
    vx = np.fft.irfft2(F_vx[..., :k_max+1])
    # vy = np.fft.ifft2(F_vy)
    return vx - uy


def mask_gen(input_shape, mask_ratio=0.5, seed=None):
    m = np.ones(input_shape)

    indices = [np.arange(i) for i in input_shape]
    I = np.meshgrid(*indices, indexing='ij')
    indices = np.array([index.reshape(-1) for index in I]).transpose(1, 0)
    num_pixel = len(indices)
    if seed is None:
        i_indices = np.random.choice(num_pixel, int(mask_ratio*num_pixel), replace=False)
    else:
        rng = np.random.RandomState(seed)
        i_indices = rng.choice(num_pixel, int(mask_ratio * num_pixel), replace=False)
    indices = indices[i_indices]
    m[tuple(indices.transpose(1, 0))] = 0
    m = m.astype(bool)
    return m


def plot_field(fields, row, col, dpi=100, q_range=None, save_fig=None):
    figsize = (col, row)
    fig, axes = plt.subplots(row, col, tight_layout=True, figsize=figsize, dpi=dpi)
    fields = fields.reshape(row, col, *fields.shape[1:])
    for i in range(row):
        for j in range(col):
            field = fields[i, j]
            pc = axes[i, j].pcolormesh(field, cmap='RdBu_r')
            if q_range is not None:
                pc.set_clim(q_range)
            axes[i, j].axis('off')
            axes[i, j].set_aspect(1)
    plt.show()
    if save_fig is not None:
        fig.savefig('./results/'+save_fig)


def cal_water_attr(t):
    attr_table = dict(
        t=[10, 20, 30, 40],
        rho=[999.7, 998.2, 995.7, 992.2],
        lamb=[0.574, 0.599, 0.618, 0.635],
        cp=[4191., 4183., 4174., 4174],
        # alpha=[20.e-6, 21.4e-6, 22.9e-6, 24.3e-6],
        mu=[1.306e-3, 1.004e-3, 0.8015e-4, 0.6533e-4],
        nu=[1.306e-6, 1.006e-6, 0.8050e-6, 0.6590e-6],
        Pr=[9.52, 7.02, 5.42, 4.31],
                      )
    t_low, t_high = attr_table['t'][0], attr_table['t'][-1]
    if t < t_low or t > t_high:
        raise ValueError(f'Input temperature out of range! Expect the input in range {t_low} to {t_high}')
    xs = attr_table['t']
    attr_t = dict()
    for key in attr_table.keys():
        v = attr_table[key]
        f = interp1d(xs, v, kind='linear')
        attr_t[key] = f(t)
    return attr_t


def sample_to_hot_wire(sample, coords, spacing, offsets=np.array([0, 0]), num_frame=10,
                       scalar=None, is_avg=True, use_para=False, weight=1.):
    # sample: b*(t*c+2)*h*w; coords: N*dim, N-points measurements of velocity; spacing: grid spacing
    device = sample.device
    indices = ((coords+offsets)/spacing).astype('int')

    if len(sample.shape) > 4:
        para = sample[:, :, num_frame:]
        sample = sample[:, :, :num_frame]
    else:
        para = sample[:, num_frame:num_frame+1]
        sample = sample[:, :num_frame]

    if scalar is not None:
        # scalar_std = torch.ones([1, len(sample[0]), 1, 1]).to(device)
        # scalar_mean = torch.zeros([1, len(sample[0]), 1, 1]).to(device)
        # scalar_std[:, :num_frame] = scalar_std[:, :num_frame]*scalar.std
        # scalar_mean[:, :num_frame] = scalar_mean[:, :num_frame]+scalar.mean
        # sample = sample*scalar.std+scalar.mean
        sample = scalar(sample)

    if len(sample.shape) > 4:
        obs = sample[:, :, :, indices[:, 0], indices[:, 1]]
        obs = torch.sqrt(obs[:, :, 0]**2+obs[:, :, 1]**2)
    else:
        obs = sample[:, :, indices[:, 0], indices[:, 1]]
        obs = torch.sqrt(obs[:, ::2]**2+obs[:, 1::2]**2)
    if is_avg:
        obs = obs.mean(1)
    if use_para:
        # obs = torch.cat([obs.reshape(len(obs), -1), weight*para.reshape(len(para), -1).mean(1)[:, None]], dim=-1)
        obs = torch.cat([obs.reshape(len(obs), -1), weight*para.reshape(len(para), -1)], dim=-1)

    return obs


def cal_rmse(gt, pred, normalize=True, reduct='sum'):
    # reduct = 'sum' or 'mean' etc.
    lib_name = np if isinstance(gt[0], np.ndarray) else torch
    reduct_fn = getattr(lib_name, reduct)
    rmse = []
    for a, b in zip(gt, pred):
        if normalize:
            coeff = 1./lib_name.sqrt(reduct_fn(a**2))
        else:
            coeff = 1.
        rmse.append(coeff*lib_name.sqrt(reduct_fn((a-b)**2)))
    return np.array(rmse) if isinstance(a, np.ndarray) else rmse


def cal_correlation(gt, pred, standardize=True, reduct='sum'):
    # standardize: whether to substract mean value of input data
    lib_name = np if isinstance(gt[0], np.ndarray) else torch
    reduct_fn = getattr(lib_name, reduct)
    cossim = []
    for a, b in zip(gt, pred):
        if standardize:
            a_mean = lib_name.mean(a)
            b_mean = lib_name.mean(b)
        else:
            a_mean = 0.
            b_mean = 0.
        a_norm = lib_name.sqrt(reduct_fn(a**2))
        b_norm = lib_name.sqrt(reduct_fn(b**2))
        cossim.append(reduct_fn((a-a_mean).reshape(-1)*(b-b_mean).reshape(-1))/(a_norm*b_norm))
    return np.array(cossim) if isinstance(a, np.ndarray) else cossim


def voronoi_interp(matrix, mask):
    """
    Completes a masked matrix using voronoi-tessellation interpolation, compatible with
    both numpy.ndarray and torch.Tensor.

    Args:
        matrix (ndarray or Tensor): 2D array containing the masked matrix.
        mask (ndarray or Tensor): 2D bool-type array or tensor.

    Returns:
        completed_matrix (ndarray or Tensor): 2D array or tensor containing the completed matrix.
    """
    
    is_tensor = torch.is_tensor(matrix)

    # Convert to NumPy if input is Tensor
    if is_tensor:
        device = matrix.device
        matrix_np = matrix.detach().cpu().numpy()
        mask_np = mask.numpy()
    else:
        matrix_np = matrix
        mask_np = mask

    # Find the indices of the masked points
    unmasked_indices = np.argwhere(mask_np == True)
    vor = Voronoi(unmasked_indices)
    values = matrix_np[unmasked_indices[:, 0], unmasked_indices[:, 1]]

    # Loop over each masked point and fill it in using the nearest-neighbor value
    grid_x, grid_y = np.meshgrid(range(matrix_np.shape[0]), range(matrix_np.shape[1]), indexing='ij')
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    tree = cKDTree(vor.points)
    _, indexes = tree.query(grid_points)
    voronoi_matrix = values[indexes].reshape(grid_x.shape)

    # Convert back to Tensor if the input was Tensor
    if is_tensor:
        return torch.from_numpy(voronoi_matrix).to(device)
    else:
        return voronoi_matrix


def Kuramoto_Sivashinsky_equation(N, h, tmax, vis, init_frame, nplt=1, M=16):
    # Initial condition and grid setup
    x = np.transpose(np.conj(np.arange(1, N+1))) / N
    a = -1
    b = 1
    # Generate samples from the Gaussian process
    # sample = gaussian_process_periodic(np.linspace(0, 1, N), init_sigma, init_amp)
    u = init_frame     # np.cos(x/16)*(1+np.sin(x/16))
    v = np.fft.fft(u)
    # scalars for ETDRK4
    k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) / 16
    L = k**2 - vis*k**4
    E = np.exp(h*L)
    E_2 = np.exp(h*L/2)
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    # main loop
    uu = np.array([u])
    tt = 0
    nmax = round(tmax/h)
    g = -0.5j*k
    for n in range(1, nmax+1):
        t = n*h
        Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = E_2*v + Q*Nv
        Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E_2*v + Q*Na
        Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E_2*a + Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        if n%nplt == 0:
            u = np.real(np.fft.ifft(v))
            uu = np.append(uu, np.array([u]), axis=0)
            tt = np.hstack((tt, t))
    return uu


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
