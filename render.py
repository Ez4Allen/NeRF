import torch
import torch.nn.functional as F

def sample_along_rays(rays_o, rays_d, near, far, S=64, stratified=True):
    B = rays_o.shape[0]
    z_lin = torch.linspace(0., 1., S, device=rays_o.device)
    z = near * (1. - z_lin) + far * z_lin
    z = z.unsqueeze(0).expand(B, S)
    if stratified:
        mids = 0.5 * (z[:, 1:] + z[:, :-1])
        lows = torch.cat([z[:, :1], mids], 1)
        highs = torch.cat([mids, z[:, -1:]], 1)
        z = lows + (highs - lows) * torch.rand_like(z)
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z[..., None]
    return z, pts

def volume_render(rgb, sigma, z, rays_d, white_bkgd=True):
    B, S, _ = rgb.shape
    deltas = z[:, 1:] - z[:, :-1]
    delta_inf = torch.full((B, 1), 1e10, device=z.device)
    deltas = torch.cat([deltas, delta_inf], 1).unsqueeze(-1)
    dist = deltas * torch.linalg.norm(rays_d, dim=-1, keepdim=True).unsqueeze(1)
    alpha = 1. - torch.exp(-sigma * dist)
    T = torch.cumprod(torch.cat([torch.ones(B,1,1, device=z.device), 1.-alpha + 1e-10], 1), 1)[:, :-1]
    weights = alpha * T
    rgb_map = torch.sum(weights * rgb, 1)
    acc_map = torch.sum(weights, 1)
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map)
    return rgb_map, acc_map, weights

def sample_pdf(bins, weights, N_importance, deterministic=False, eps=1e-5):
    w = (weights + eps)
    pdf = w / torch.sum(w, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if deterministic:
        u = torch.linspace(0., 1., N_importance, device=bins.device)
        u = u.unsqueeze(0).expand(bins.shape[0], N_importance)
    else:
        u = torch.rand(bins.shape[0], N_importance, device=bins.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    cdf_b = torch.gather(cdf, 1, below)
    cdf_a = torch.gather(cdf, 1, above)
    bins_b = torch.gather(bins, 1, torch.clamp(below - 1, min=0))
    bins_a = torch.gather(bins, 1, torch.clamp(above - 1, min=0))
    denom = (cdf_a - cdf_b).clamp_min(eps)
    t = (u - cdf_b) / denom
    return bins_b + t * (bins_a - bins_b)
