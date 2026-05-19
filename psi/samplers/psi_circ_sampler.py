"""SFM sampler with CIRCULATION-constrained tangent residual psi.

Supports two modes:
  (1) Deterministic inference (default): vel = v_sfm + psi_circ; exp_map step.
  (2) Stochastic rollout (REINFORCE):    vel = v_sfm + psi_circ; exp_map with
      additional Brownian-tangent noise; per-step trajectory recorded for
      off-line policy-gradient computation.

Reuses the well-tested SFMSampler.step infrastructure (precision, slicing,
prefix projection, last-step decode) and only adds the noise + recording.
"""
import math
import torch
import utils
from samplers import SFMSampler, SFMContext
from psi.nets.psi_circ_net import (CirculationPsiNet, project_tangent,
                                   project_out_score,
                                   spherical_score_from_xpred)


class SFMSamplerWithPsiCirc(SFMSampler):
    def __init__(self, psi_net=None, psi_scale=1.0,
                 sigma_explore=0.0, record_trajectory=False, **kwargs):
        super().__init__(**kwargs)
        self.psi_net = psi_net
        self.psi_scale = float(psi_scale)
        self.sigma_explore = float(sigma_explore)
        self.record_trajectory = bool(record_trajectory)
        self.trajectory = []   # list of per-step dicts (when recording)

    def attach_psi(self, psi_net):
        self.psi_net = psi_net

    def set_psi_scale(self, scale):
        self.psi_scale = float(scale)

    def set_sigma_explore(self, sigma):
        self.sigma_explore = float(sigma)

    def reset_trajectory(self):
        self.trajectory = []

    def step(self, model, state):
        num_steps = len(state.t_schedule) - 1
        is_last_step = (state.step_idx == num_steps - 1)

        _, alpha_t = model.noise(state.t_schedule[state.step_idx])
        sigma_t = model._sigma_from_alphat(alpha_t).reshape(-1, 1)
        context = SFMContext(temperature=self.temperature)
        log_p = model.forward(xt=state.xt, sigma=sigma_t, context=context)
        if self.use_float64:
            log_p = log_p.to(torch.float64)
        state.nfe += 1

        if self.p_nucleus != 1.0 or self.top_k != -1:
            log_p = utils.top_k_top_p_filtering(
                log_p, top_k=self.top_k, top_p=self.p_nucleus).log_softmax(-1)
        if is_last_step:
            return self._last_step_decode(state, log_p)

        log_p_window = log_p[:, state.start_idx:]
        E = utils.sphere_normalize(
            model.backbone.sphere_embed.weight.detach())
        x = state.xt[:, state.start_idx:].to(E)
        if self.slerp_float64:
            E = E.to(torch.float64)
            x = x.to(torch.float64)
        if self.top_k_velocity > 0:
            log_p_v, E = self._select_topk(log_p_window, E, self.top_k_velocity)
        else:
            log_p_v = log_p_window

        v_sfm = self._compute_velocity(x, E, log_p_v)

        # ---- PSI INJECTION (circulation variant) ----
        psi_val = None
        score = None
        c = None
        if self.psi_net is not None:
            B = x.shape[0]
            alpha_b = alpha_t.reshape(-1).to(x.dtype)
            if alpha_b.numel() == 1:
                alpha_b = alpha_b.expand(B)
            if state.prefix_embeds is not None:
                c = state.prefix_embeds.to(x.dtype).mean(dim=1)
            else:
                c = torch.zeros((B, x.shape[-1]),
                                device=x.device, dtype=x.dtype)
            # Spherical score from the predicted clean point (top-k aware)
            p = log_p_v.exp()
            if E.ndim == 2:
                x_pred = torch.einsum('blv,vd->bld', p, E)
            else:
                x_pred = torch.einsum('blk,blkd->bld', p, E)
            x_pred = utils.sphere_normalize(x_pred)
            score = spherical_score_from_xpred(x, x_pred)

            psi_dtype = next(self.psi_net.parameters()).dtype
            with torch.no_grad():
                psi_val = self.psi_net(
                    x.to(psi_dtype), alpha_b.to(psi_dtype),
                    c.to(psi_dtype), score.to(psi_dtype))
            psi_val = psi_val.to(x.dtype)
            # Belt-and-suspenders re-project
            psi_val = project_tangent(psi_val, x)
            psi_val = project_out_score(psi_val, score)
            if self.psi_scale != 1.0:
                psi_val = psi_val * self.psi_scale
            vel = v_sfm + psi_val
        else:
            vel = v_sfm
        # ----------------------------------------------

        dt = self._get_step_size(model, state)
        drift = dt * vel

        # Brownian-tangent noise injection
        if self.sigma_explore > 0:
            bern = torch.randn_like(x)
            xi = project_tangent(bern, x)
            noise = self.sigma_explore * math.sqrt(abs(float(dt))) * xi
            step_taken = drift + noise
        else:
            xi = None
            step_taken = drift

        x_new = utils.exp_map(x, step_taken, self.eps)
        state.xt[:, state.start_idx:] = x_new.to(state.xt.dtype)
        self._project_prefix(
            state.xt, state.prefix_embeds, state.prefix_lengths)

        # Record for REINFORCE
        if self.record_trajectory:
            self.trajectory.append({
                'z':         x.detach().clone(),
                'v_sfm':     v_sfm.detach(),
                'psi':       psi_val.detach() if psi_val is not None else None,
                'score':     score.detach() if score is not None else None,
                'c':         c.detach() if c is not None else None,
                'alpha':     (alpha_t.detach() if torch.is_tensor(alpha_t)
                              else torch.tensor(float(alpha_t),
                                                device=x.device)),
                'dt':        float(dt),
                'drift':     drift.detach(),
                'xi':        xi.detach() if xi is not None else None,
                'step':      step_taken.detach(),
                'start_idx': state.start_idx,
            })

        state.step_idx += 1
        return state
