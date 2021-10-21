import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .base_nerf_hypernet import BaseNerfHypernet


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, n_heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = inner_dim
        self.scale = head_dim ** -0.5

        self.softmax = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, fr, to):
        # fr: (B, L, D)
        B, L_fr, L_to = fr.shape[0], fr.shape[1], to.shape[1]
        q = self.to_q(fr).view(B, L_fr, self.n_heads, self.head_dim).transpose(1, 2) # (B, heads, L_fr, head_dim)
        kv = self.to_kv(to).view(B, L_to, self.n_heads, self.head_dim * 2).transpose(1, 2)
        k, v = kv[..., :self.head_dim], kv[..., self.head_dim:] # (B, heads, L_to, head_dim)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (B, heads, L_fr, L_to)
        attn = self.softmax(dots)
        out = torch.matmul(attn, v) # (B, heads, L_fr, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L_fr, -1)
        return self.to_out(out)


@register('nh-trans_learner')
class NHTransLearner(BaseNerfHypernet):

    def __init__(self, n_steps, patch_size, imgt_dim, n_heads, head_dim, ff_dim, use_viewdirs, dropout=0.):
        super().__init__(use_viewdirs=use_viewdirs)
        self.n_steps = n_steps

        self.patch_size = patch_size
        self.imgt_prefc = nn.Linear(patch_size**2 * 9, imgt_dim)
        self.imgt_norm = nn.LayerNorm(imgt_dim)
        self.imgt_att = nn.ModuleList([Attention(imgt_dim, n_heads, head_dim, dropout=dropout) for _ in range(n_steps)])
        self.imgt_ff = nn.ModuleList([FeedForward(imgt_dim, ff_dim, dropout=dropout) for _ in range(n_steps)])

        self.wt_init = nn.ParameterDict()
        self.wt_norm = nn.ModuleDict()
        self.wt_self_att = nn.ModuleDict()
        self.trans_fc = nn.ModuleDict()
        self.wt_crs_att = nn.ModuleDict()
        self.wt_ff = nn.ModuleDict()
        for name, shape in self.hypo_nerf.params_shape.items():
            n, d = shape[1], shape[0]
            self.wt_init[name] = nn.Parameter(torch.randn(n, d))
            self.wt_norm[name] = nn.LayerNorm(d)
            self.wt_self_att[name] = nn.ModuleList([Attention(d, n_heads // 2, head_dim // 2, dropout=dropout) for _ in range(n_steps)])
            self.trans_fc[name] = nn.ModuleList([nn.Linear(imgt_dim, d) for _ in range(n_steps)])
            self.wt_crs_att[name] = nn.ModuleList([Attention(d, n_heads // 2, head_dim // 2, dropout=dropout) for _ in range(n_steps)])
            self.wt_ff[name] = nn.ModuleList([FeedForward(d, ff_dim // 4, dropout=dropout) for _ in range(n_steps)])

    def generate_params(self, rays_o, rays_d, imgs):
        """
            imgs: shape (B, N, 3, H, W)
            rays_o, rays_d: shape (B, N, H, W, 3)
        """
        B, N, _, H, W = imgs.shape

        # Image token generation line
        x = torch.cat([imgs,
                       rays_o.permute(0, 1, 4, 2, 3),
                       rays_d.permute(0, 1, 4, 2, 3)], dim=2).view(B * N, -1, H, W)
        P = self.patch_size
        x = F.unfold(x, P, stride=P) # (B * N, P * P * 9, (H // P) * (W // P))
        x = x.permute(0, 2, 1).contiguous().view(B, N * (H // P) * (W // P), P * P * 9)
        x = self.imgt_prefc(x)
        imgt_res = []
        for i in range(self.n_steps):
            _ = self.imgt_norm(x); x = x + self.imgt_att[i](_, _)
            x = x + self.imgt_ff[i](self.imgt_norm(x))
            imgt_res.append(x)

        # Trans learner line
        params = dict()
        for name, shape in self.hypo_nerf.params_shape.items():
            fr = self.wt_init[name].unsqueeze(0).expand(B, -1, -1)
            params[name] = fr.transpose(-1, -2); continue
            norm = self.wt_norm[name]
            for i in range(self.n_steps):
                _ = norm(fr); fr = fr + self.wt_self_att[name][i](_, _)
                to = self.trans_fc[name][i](imgt_res[i])
                fr = fr + self.wt_crs_att[name][i](norm(fr), norm(to))
                fr = fr + self.wt_ff[name][i](norm(fr))
            params[name] = fr.transpose(-1, -2)

        return params
