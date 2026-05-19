"""SFM sampler that injects a learned tangent residual psi to the velocity.

At each step:
  vel = sfm_compute_velocity(...)               # tangent at x
  psi_tan = psi_phi(x, tau, c)                  # already tangent (network projects)
  vel_total = vel + psi_tan
  x_new = exp_map(x, dt * vel_total)            # geodesic step on sphere

c (the conditioning) is the mean-pooled prompt sphere embedding for
prefix-conditioned generation. For unconditional generation, c is a
zero vector.
"""
import torch
import utils
from samplers import SFMSampler, SFMContext, sfm_compute_velocity, sfm_step_size
from psi.nets.psi_sphere_net import SphereSolenoidalNet, project_tangent


class SFMSamplerWithPsi(SFMSampler):
    """SFMSampler subclass: same velocity computation, but adds psi(x, tau, c)
    to vel before the exp_map step. Setting psi=None or its params=0 reduces
    to vanilla SFMSampler (sanity)."""

    def __init__(self, psi_net=None, psi_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.psi_net = psi_net   # nn.Module or None
        self.psi_scale = float(psi_scale)

    def attach_psi(self, psi_net):
        self.psi_net = psi_net

    def set_psi_scale(self, scale):
        self.psi_scale = float(scale)

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

        vel = self._compute_velocity(x, E, log_p_v)

        # ---- PSI INJECTION ----
        if self.psi_net is not None:
            # Pass alpha_t (the [0,1] noise fraction) to psi for consistency
            # with the training script — schedule-time t differs across noise
            # configs, alpha_t is canonical.
            B = x.shape[0]
            tau_b = alpha_t.reshape(-1).to(x.dtype)
            if tau_b.numel() == 1:
                tau_b = tau_b.expand(B)
            # Build conditioning c: mean-pooled prompt sphere embedding,
            # or zero vector if no prefix.
            if state.prefix_embeds is not None:
                # prefix_embeds: (B, P, d) sphere embeddings of the prompt
                c = state.prefix_embeds.to(x.dtype).mean(dim=1)
            else:
                c = torch.zeros((B, x.shape[-1]), device=x.device, dtype=x.dtype)
            # psi_net params may be float32 even when state is float64
            # (SFM uses slerp_float64). Run psi in net's dtype, cast back.
            psi_dtype = next(self.psi_net.parameters()).dtype
            with torch.no_grad():
                psi_val = self.psi_net(
                    x.to(psi_dtype), tau_b.to(psi_dtype), c.to(psi_dtype))
            psi_val = psi_val.to(x.dtype)
            # Belt-and-suspenders: re-project to tangent (numerical drift)
            psi_val = project_tangent(psi_val, x)
            if self.psi_scale != 1.0:
                psi_val = psi_val * self.psi_scale
            vel = vel + psi_val.to(vel.dtype)
        # ------------------------

        dt = self._get_step_size(model, state)
        x_new = utils.exp_map(x, dt * vel, self.eps)
        state.xt[:, state.start_idx:] = x_new.to(state.xt.dtype)
        self._project_prefix(
            state.xt, state.prefix_embeds, state.prefix_lengths)
        state.step_idx += 1
        return state
