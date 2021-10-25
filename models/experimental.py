import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.nvs.base_nvs_hypernet import BaseNvsHypernet


@register('nvsh-transformer_modfc')
class NvshTransformerModfc(BaseNvsHypernet):

    def __init__(self, patch_size, d_model, nhead, dim_feedforward, dropout, num_layers, n_groups, use_viewdirs, num_modfc):
        super().__init__(use_viewdirs=use_viewdirs)
        self.patch_size = patch_size
        img_channels = 3

        # Pre FC
        self.patch_prefc = nn.Linear(self.patch_size * self.patch_size * (img_channels + 6), d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Base params
        self.base_params = dict()
        for name, shape in self.hyponet.params_shape.items():
            weight = torch.empty(shape[1], shape[0] - 1)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

            bias = torch.empty(shape[1], 1)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

            p = nn.Parameter(torch.cat([weight, bias], dim=1).t().detach())
            self.base_params[name] = p
            self.register_parameter(f'base_params_{name}', p)

        # Define modulate vectors
        self.num_modfc = num_modfc
        self.ptoken_rng = dict()
        self.ptoken_postfc = nn.ModuleDict()
        n_ptokens = 0
        for name, shape in self.hyponet.params_shape.items():
            if name == 'wb_rgbsigma' or int(name[-1]) >= self.num_modfc:
                continue
            g = min(n_groups, shape[1]) # group or all
            assert shape[1] % g == 0
            self.ptoken_postfc[name] = nn.Linear(d_model, shape[0] - 1)
            self.ptoken_rng[name] = (n_ptokens, n_ptokens + g)
            n_ptokens += g
        self.ptokens = nn.Parameter(torch.randn(n_ptokens, d_model))

    def generate_params(self, rays_o, rays_d, imgs):
        """
            imgs: shape (B, N, 3, H, W)
            rays_o, rays_d: shape (B, N, H, W, 3)
        """
        B, N, _, H, W = imgs.shape

        # Pre FC to tokens
        x = torch.cat([imgs,
                       rays_o.permute(0, 1, 4, 2, 3),
                       rays_d.permute(0, 1, 4, 2, 3)], dim=2).view(B * N, -1, H, W)
        P = self.patch_size
        x = F.unfold(x, P, stride=P) # (B * N, P * P * 9, (H // P) * (W // P))
        x = x.permute(0, 2, 1).contiguous().view(B, N * (H // P) * (W // P), -1)
        x = x.permute(1, 0, 2) # (n_patches, B, P * P * 9)
        x = self.patch_prefc(x)

        # Feed transformer with param tokens
        n_ptokens = len(self.ptokens)
        ptokens = self.ptokens.unsqueeze(1).expand(-1, B, -1)
        x = torch.cat([ptokens, x], dim=0) # (n_ptokens + n_patches, B, D)
        outp = self.transformer(x)[:n_ptokens] # (n_ptokens, B, D)

        # Translate to params
        params = dict()
        for name, shape in self.hyponet.params_shape.items():
            wb = self.base_params[name].unsqueeze(0).expand(B, -1, -1)

            if name == 'wb_rgbsigma' or int(name[-1]) >= self.num_modfc:
                pass
            else:
                ql, qr = self.ptoken_rng[name]
                x = self.ptoken_postfc[name](outp[ql: qr]) # (g, B, (shape[0] - 1)); g = min(shape[1], n_groups)
                x = x.permute(1, 2, 0) # (B, (shape[0] - 1), g)
                w, b = wb[:, :-1, :], wb[:, -1:, :]
                w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)
                wb = torch.cat([w, b], dim=1)

            params[name] = wb

        return params


####


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


@register('nvs_hypernet-trans_learner')
class NvsHypernetTransLearner(BaseNvsHypernet):

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
        for name, shape in self.hyponet.params_shape.items():
            n, d = shape[1], shape[0]
            self.wt_init[name] = nn.Parameter(torch.empty(n, d))
            nn.init.kaiming_uniform_(self.wt_init[name], a=math.sqrt(5))
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
        for name, shape in self.hyponet.params_shape.items():
            fr = self.wt_init[name].unsqueeze(0).expand(B, -1, -1)
            #params[name] = fr.transpose(-1, -2); continue
            norm = self.wt_norm[name]
            for i in range(self.n_steps):
                _ = norm(fr); fr = fr + self.wt_self_att[name][i](_, _)
                to = self.trans_fc[name][i](imgt_res[i]) ## [i]
                fr = fr + self.wt_crs_att[name][i](norm(fr), norm(to))
                fr = fr + self.wt_ff[name][i](norm(fr))
            params[name] = fr.transpose(-1, -2)

        return params
