import torch


def get_rays_batch(h, w, f, poses):
    """
    Args:
        h, w, f: height, width, focal length
        poses: shape (n, 3, 4), transform matrix
    Returns:
        rays_o, rays_d: shape (n, h, w, 3)
    """
    n = poses.shape[0]
    yv, xv = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float())
    xv, yv = xv + .5, yv + .5 ## not ori, but why dont we want centering
    dirs = torch.stack([(xv - w / 2) / f, -(yv - h / 2) / f, -torch.ones(h, w)], dim=-1) # (h, w, 3)
    rays_d = (dirs.view(1, h, w, 1, 3) * poses[:, :, :3].view(n, 1, 1, 3, 3)).sum(dim=-1)
    rays_o = poses[:, :, -1].view(n, 1, 1, 3).expand(n, h, w, 3)
    return rays_o, rays_d


def get_rays(h, w, f, pose):
    rays_o, rays_d = get_rays_batch(h, w, f, pose.unsqueeze(0))
    return rays_o[0], rays_d[0]


def render_rays(nerf, rays_o, rays_d, params=None, near=2, far=6, n_samples=128, use_viewdirs=False,
                rand=False, allret=False):
    """
    Args:
        rays_o, rays_d: shape (..., 3)
        params: optional for hypernets. If given, there is batch dim.
    Returns:
        rgb_ret: shape (..., 3)
    """
    device = rays_o.device
    query_shape = rays_o.shape[:-1]

    rays_o = rays_o.view(-1, 3) # (n_rays, 3)
    rays_d = rays_d.view(-1, 3)
    n_rays = rays_o.shape[0]

    # Compute 3D query points
    z_vals = torch.linspace(near, far, n_samples, device=device) # (n_samples,)
    if rand:
        z_vals = z_vals + torch.rand(n_rays, n_samples, device=device) * (far - near) / n_samples # (n_rays, n_samples)
    else:
        z_vals = z_vals.unsqueeze(0).expand(n_rays, n_samples) # (n_rays, n_samples)
    coords = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # (n_rays, n_samples, 3)
    viewdirs = rays_d.unsqueeze(-2).expand(-1, n_samples, -1)

    # Run network
    if params is None:
        if use_viewdirs:
            model_outp = nerf(coords, viewdirs=viewdirs)
        else:
            model_outp = nerf(coords)
    else:
        B = query_shape[0] # query_shape[0] should be batch
        coords = coords.view(B, n_rays * n_samples // B, 3)
        viewdirs = viewdirs.contiguous().view(B, n_rays * n_samples // B, 3)
        if use_viewdirs:
            model_outp = nerf(coords, viewdirs=viewdirs, params=params)
        else:
            model_outp = nerf(coords, params=params)
        model_outp = model_outp.view(n_rays, n_samples, 4)
    rgb = model_outp[..., :3] # (n_rays, n_samples, 3), sigmoid-ed
    sigma = model_outp[..., 3] # (n_rays, n_samples), relu-ed

    # Do volume rendering
    dists = torch.cat([
        z_vals[:, 1:] - z_vals[:, :-1],
        1e-3 * torch.ones_like(z_vals[:, -1:]),
    ], dim=-1) # (n_rays, n_samples)
    trans = torch.exp(-sigma * dists)
    alpha = 1 - trans
    trans = torch.minimum(trans + 1e-10, torch.tensor(1, dtype=torch.float, device=device))
    trans = torch.cat([torch.ones_like(trans[..., :1]), trans[..., :-1]], dim=-1)
    weights = alpha * torch.cumprod(trans, dim=-1)

    rgb_ret = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    acc_ret = torch.sum(weights, dim=-1)
    rgb_ret = rgb_ret + (1 - acc_ret.unsqueeze(-1)) # white background
    if not allret:
        return rgb_ret.view(*query_shape, -1)

    depth_ret = torch.sum(weights * z_vals, -1)
    return rgb_ret.view(*query_shape, -1), depth_ret.view(*query_shape, -1), acc_ret.view(*query_shape, -1)
