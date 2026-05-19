"""Circulation-constrained psi network for B'.

Hard constraint: psi LIVES in the circulation subspace at each point z,
where 'circulation' means orthogonal to the (tangent) score s_tau.

The intuition: score points in the direction of increasing log-density of
the frozen flow's marginal p_tau. Adding a tangent component along s_tau
moves trajectories *toward / away from* the high-density region — i.e.
transport. The orthogonal complement (circulation) re-routes trajectories
WITHIN the same level set, which is the closest first-order analog of the
divergence-free / measure-invariant subspace.

So we project out the score component AFTER projecting to the sphere
tangent. Both projections are HARD (architectural), not soft penalties.

psi(z, tau, c, s_tau) =
    psi_raw      = MLP(z, tau, c)
    psi_sphere   = psi_raw  - <psi_raw, z>   z              # tangent at z
    psi_circ     = psi_sphere - <psi_sphere, s_tau>/|s_tau|^2 * s_tau

By construction:  psi_circ . z = 0   AND   psi_circ . s_tau = 0.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embed(tau, dim=64, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=tau.device) / half)
    args = tau[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def project_tangent(v, x):
    """v - (v.x) x  per position."""
    return v - (v * x).sum(dim=-1, keepdim=True) * x


def project_out_score(v_tan, score, eps=1e-8):
    """v_tan - <v_tan, score>/|score|^2 * score  per position.
    Both v_tan and score must already be tangent at the same point.
    """
    s2 = (score * score).sum(dim=-1, keepdim=True).clamp(min=eps)
    coef = (v_tan * score).sum(dim=-1, keepdim=True) / s2
    return v_tan - coef * score


class CirculationPsiNet(nn.Module):
    """psi(z, tau, c, score) -> circulation tangent at z.
    Same MLP backbone as SphereSolenoidalNet; output is hard-projected
    to the circulation subspace.
    """
    def __init__(self, d, hidden=512, tau_dim=64, cond_dim=256):
        super().__init__()
        self.d = d
        self.tau_dim = tau_dim
        self.cond_dim = cond_dim
        self.cond_fc1 = nn.Linear(tau_dim + d, cond_dim)
        self.cond_fc2 = nn.Linear(cond_dim, cond_dim)
        self.fc1 = nn.Linear(d, hidden)
        self.film1_scale = nn.Linear(cond_dim, hidden)
        self.film1_shift = nn.Linear(cond_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.film2_scale = nn.Linear(cond_dim, hidden)
        self.film2_shift = nn.Linear(cond_dim, hidden)
        self.fc_out = nn.Linear(hidden, d)
        # Zero-init: psi == 0 at init
        for m in (self.film1_scale, self.film1_shift,
                  self.film2_scale, self.film2_shift, self.fc_out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _raw(self, x, tau, c):
        tau_emb = sinusoidal_embed(tau, dim=self.tau_dim)
        cond = torch.cat([tau_emb, c], dim=-1)
        cond = F.silu(self.cond_fc1(cond))
        cond = F.silu(self.cond_fc2(cond))
        h = self.fc1(x)
        s1 = self.film1_scale(cond)[:, None, :]
        sh1 = self.film1_shift(cond)[:, None, :]
        h = F.silu(h * (1.0 + s1) + sh1)
        h = self.fc2(h)
        s2 = self.film2_scale(cond)[:, None, :]
        sh2 = self.film2_shift(cond)[:, None, :]
        h = F.silu(h * (1.0 + s2) + sh2)
        return self.fc_out(h)

    def forward(self, x, tau, c, score):
        # x, score: (B, L, d). score is assumed tangent at x.
        psi_raw = self._raw(x, tau, c)
        psi_sphere = project_tangent(psi_raw, x)
        psi_circ = project_out_score(psi_sphere, score)
        return psi_circ


def spherical_score_from_xpred(z, x_pred):
    """Tangent at z, pointing toward (the projection of) x_pred."""
    return x_pred - (z * x_pred).sum(dim=-1, keepdim=True) * z
