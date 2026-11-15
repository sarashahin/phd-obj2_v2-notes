#!/usr/bin/env python3
"""
obj2_model.py  (obj2_v2, stage F, 8 Nov 2026; new model file agreed by Sara on 8 Nov)

Conditional denoising U-Net and diffusion for the 6-year stack of one species (target P1), on the 24 x 24 canvas of
stage E. One file, used by training, validation, inference and BNM; nothing from the old model code (no graph, no
temporal encoder, no growth field; root causes F4-F9, R6, R7, R21).

    input   x_t (6 yearly maps, noised, zero outside the window) + the 22 stage E channels = 28 channels;
            the diffusion step and the 2 stage E numbers enter every block through a learned scale and shift
    output  the predicted noise for the 6 yearly maps
    levels  24 x 24 (base channels), 12 x 12 (2 x base), 6 x 6 (3 x base, with self-attention); zero padding, no wrap
    diffusion  DDPM training with a cosine noise schedule, 1,000 steps (Ho et al. 2020; Nichol and Dhariwal 2021);
            sampling by DDIM with eta 0 (Song et al. 2021), so a fixed starting noise gives a fixed sample; the maps
            are held at 0 outside the window at every step, and the loss counts window cells only

References: Ho J, Jain A, Abbeel P (2020) Denoising diffusion probabilistic models. NeurIPS 33: 6840-6851.
Nichol AQ, Dhariwal P (2021) Improved denoising diffusion probabilistic models. ICML 2021, PMLR 139: 8162-8171.
Song J, Meng C, Ermon S (2021) Denoising diffusion implicit models. ICLR 2021. arXiv 2010.02502.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_COND, N_NUM, N_OUT, MASK_CH = 22, 2, 6, 21


def groups(c):
    return next(g for g in (8, 4, 2, 1) if c % g == 0)


def step_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(groups(cin), cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(groups(cout), cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.e(F.silu(e)).chunk(2, 1)
        h = self.n2(h) * (1 + sc[..., None, None]) + sh[..., None, None]
        return self.c2(F.silu(h)) + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(groups(c), c)
        self.a = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.n(x).flatten(2).transpose(1, 2)
        return x + self.a(s, s, s, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, base=48, emb=192):
        super().__init__()
        c0, c1, c2 = base, 2 * base, 3 * base
        self.emb_dim = emb
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.n_mlp = nn.Sequential(nn.Linear(N_NUM, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv2d(N_OUT + N_COND, c0, 3, padding=1)
        self.d0 = nn.ModuleList([ResBlock(c0, c0, emb), ResBlock(c0, c0, emb)])
        self.down0 = nn.Conv2d(c0, c0, 3, stride=2, padding=1)
        self.d1 = nn.ModuleList([ResBlock(c0, c1, emb), ResBlock(c1, c1, emb)])
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2 = nn.ModuleList([ResBlock(c1, c2, emb), ResBlock(c2, c2, emb)])
        self.a2 = Attn(c2)
        self.m1, self.ma, self.m2 = ResBlock(c2, c2, emb), Attn(c2), ResBlock(c2, c2, emb)
        self.u2 = nn.ModuleList([ResBlock(2 * c2, c2, emb), ResBlock(c2, c2, emb)])
        self.up2 = nn.Conv2d(c2, c1, 3, padding=1)
        self.u1 = nn.ModuleList([ResBlock(2 * c1, c1, emb), ResBlock(c1, c1, emb)])
        self.up1 = nn.Conv2d(c1, c0, 3, padding=1)
        self.u0 = nn.ModuleList([ResBlock(2 * c0, c0, emb), ResBlock(c0, c0, emb)])
        self.out_n, self.out = nn.GroupNorm(groups(c0), c0), nn.Conv2d(c0, N_OUT, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, xt, cond, t, nums):
        e = self.t_mlp(step_embedding(t, self.emb_dim)) + self.n_mlp(nums)
        h = self.inp(torch.cat([xt, cond], 1))
        for b in self.d0:
            h = b(h, e)
        s0 = h
        h = self.down0(h)
        for b in self.d1:
            h = b(h, e)
        s1 = h
        h = self.down1(h)
        for b in self.d2:
            h = b(h, e)
        h = self.a2(h); s2 = h
        h = self.m2(self.ma(self.m1(h, e)), e)
        h = torch.cat([h, s2], 1)
        for b in self.u2:
            h = b(h, e)
        h = self.up2(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = torch.cat([h, s1], 1)
        for b in self.u1:
            h = b(h, e)
        h = self.up1(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = torch.cat([h, s0], 1)
        for b in self.u0:
            h = b(h, e)
        return self.out(F.silu(self.out_n(h)))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


class Diffusion:
    def __init__(self, T=1000, s=0.008, device="cpu"):
        x = torch.linspace(0, T, T + 1, dtype=torch.float64)
        f = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
        ab = f / f[0]
        betas = torch.clamp(1 - ab[1:] / ab[:-1], max=0.999)
        self.T = T
        self.ab = torch.cumprod(1 - betas, 0).float().to(device)       # alpha-bar for steps 0 .. T-1

    def to(self, device):
        self.ab = self.ab.to(device)
        return self

    def q_sample(self, x0, t, noise):
        a = self.ab[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model, x0, cond, nums, gen):
        """Mean squared error of the predicted noise over window cells of the 6 years."""
        b = len(x0)
        m = cond[:, MASK_CH:MASK_CH + 1]
        t = torch.randint(0, self.T, (b,), generator=gen, device=x0.device)
        noise = torch.randn(x0.shape, generator=gen, device=x0.device)
        xt = self.q_sample(x0, t, noise) * m
        eps = model(xt, cond, t, nums)
        return (((eps - noise) ** 2) * m).sum() / (m.sum() * x0.shape[1])

    @torch.no_grad()
    def ddim(self, model, cond, nums, noise, steps=50):
        """Deterministic DDIM (eta 0) from the given starting noise; returns the final estimate of the maps in [-1, 1]."""
        m = cond[:, MASK_CH:MASK_CH + 1]
        ts = torch.linspace(self.T - 1, 0, steps, device=cond.device).round().long()
        x = noise * m
        x0 = x
        for i, t in enumerate(ts):
            a = self.ab[t]
            tt = torch.full((len(x),), int(t), device=cond.device, dtype=torch.long)
            eps = model(x, cond, tt, nums)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1) * m
            if i + 1 < len(ts):
                ap = self.ab[ts[i + 1]]
                eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
                x = (ap.sqrt() * x0 + (1 - ap).sqrt() * eps) * m
        return x0