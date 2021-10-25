import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, dim, ff_hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, n_heads, head_dim, dropout=0.):
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


class TransformerEncoder(nn.Module):

    def __init__(self, dim, n_layers, n_heads, head_dim, ff_hidden_dim, dropout=0.):
        super().__init__()
        self.n_layers = n_layers
        self.norm = nn.LayerNorm(dim)
        self.attns = nn.ModuleList([Attention(dim, n_heads, head_dim, dropout=dropout) for _ in range(n_layers)])
        self.ffs = nn.ModuleList([FeedForward(dim, ff_hidden_dim, dropout=dropout) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            y = self.norm(x)
            x = x + self.attns[i](y, y)
            y = self.norm(x)
            x = x + self.ffs[i](y)
        return x
