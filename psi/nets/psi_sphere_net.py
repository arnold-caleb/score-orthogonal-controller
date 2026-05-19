"""SolenoidalNet on the hypersphere.

Output is *tangent* to the sphere at the current state x by construction
(orthogonally projected: psi_tan = psi - (psi . x) x). Per-position MLP
with FiLM conditioning on (tau, c). Zero-initialized output so psi == 0
at init -> ELF/SFM behavior is unchanged at init.

c is the prompt summary (we'll use the mean-pooled prompt sphere
embedding) so steering can be prompt-conditional.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embed(tau: torch.Tensor, dim: int = 64,
                     max_period: float = 10000.0) -> torch.Tensor:
    """tau: (B,) -> (B, dim) sinusoidal embedding."""
    half = dim // 2
    device = tau.device
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=device) / half)
    args = tau[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def project_tangent(psi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Project psi onto the tangent space of the sphere at x (per position).
    Both shapes (B, L, d). Returns psi - (psi . x) x  (per-position).
    """
    # x is assumed already sphere-normalized.
    dot = (psi * x).sum(dim=-1, keepdim=True)
    return psi - dot * x


class SphereSolenoidalNet(nn.Module):
    """psi_phi(x, tau, c) on the sphere.
    x: (B, L, d) sphere state. tau: (B,). c: (B, d).
    Output: (B, L, d) tangent vector at x.
    """
    def __init__(self, d: int, hidden: int = 512, tau_dim: int = 64,
                 cond_dim: int = 256):
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.tau_dim = tau_dim
        self.cond_dim = cond_dim
        # Conditioning trunk on (tau, c)
        self.cond_fc1 = nn.Linear(tau_dim + d, cond_dim)
        self.cond_fc2 = nn.Linear(cond_dim, cond_dim)
        # Main net on x with FiLM modulation
        self.fc1 = nn.Linear(d, hidden)
        self.film1_scale = nn.Linear(cond_dim, hidden)
        self.film1_shift = nn.Linear(cond_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.film2_scale = nn.Linear(cond_dim, hidden)
        self.film2_shift = nn.Linear(cond_dim, hidden)
        self.fc_out = nn.Linear(hidden, d)
        # Zero-init the FiLM scales/shifts and the output, so psi == 0 at init.
        for m in (self.film1_scale, self.film1_shift,
                  self.film2_scale, self.film2_shift, self.fc_out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        assert D == self.d, f"x.last_dim={D} != self.d={self.d}"
        tau_emb = sinusoidal_embed(tau, dim=self.tau_dim)        # (B, tau_dim)
        cond = torch.cat([tau_emb, c], dim=-1)                    # (B, tau_dim+d)
        cond = F.silu(self.cond_fc1(cond))
        cond = F.silu(self.cond_fc2(cond))                        # (B, cond_dim)

        h = self.fc1(x)                                           # (B, L, hidden)
        s1 = self.film1_scale(cond)[:, None, :]                   # (B, 1, hidden)
        sh1 = self.film1_shift(cond)[:, None, :]
        h = h * (1.0 + s1) + sh1
        h = F.silu(h)

        h = self.fc2(h)
        s2 = self.film2_scale(cond)[:, None, :]
        sh2 = self.film2_shift(cond)[:, None, :]
        h = h * (1.0 + s2) + sh2
        h = F.silu(h)

        psi_raw = self.fc_out(h)                                  # (B, L, d)
        # Project to tangent at x (preserves sphere geometry).
        return project_tangent(psi_raw, x)


def hutchinson_tangent_divergence(
        psi_apply, x: torch.Tensor, tau: torch.Tensor, c: torch.Tensor,
        rademacher: bool = True) -> torch.Tensor:
    """Estimate tangent divergence  div_S(psi)  via Hutchinson with
    Rademacher probes, projected to the tangent space at x:

        div_S(psi)(x) = trace((I - x x^T) d psi / dx)

    psi_apply: callable(x, tau, c) -> (B, L, d) tangent vector.
    Returns per-sample sum: shape (B,).
    """
    # Sample probes on the sphere tangent: bern * project_tangent(bern, x).
    if rademacher:
        bern = (torch.randint(0, 2, x.shape, device=x.device,
                              dtype=x.dtype) * 2.0 - 1.0)
    else:
        bern = torch.randn_like(x)
    v = project_tangent(bern, x)
    # jvp via finite difference (cheap, since psi_apply is a small MLP)
    eps = 1e-3
    with torch.enable_grad():
        x_req = x.detach().clone().requires_grad_(True)
        psi_val = psi_apply(x_req, tau, c)
        # Compute (v * dPsi/dx . v).sum over (L, d) per batch
        # Use jvp via grad: g = grad((v * psi).sum(), x)  →  v^T (dpsi/dx)
        # Then div_estimate ≈ (g * v).sum(axis=(-1,-2))
        vp = (v * psi_val).sum()
        g, = torch.autograd.grad(vp, x_req, create_graph=False,
                                 retain_graph=False)
    return (g * v).sum(dim=(-1, -2))
