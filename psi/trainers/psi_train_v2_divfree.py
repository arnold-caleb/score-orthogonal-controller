"""Variant B — A + tangent-divergence penalty (circulation-aligned residual).

Same supervised target-matching as variant A, plus a penalty pulling psi
toward the measure-divergence-free subspace on the sphere:

    div_S(psi)  +  psi . s_tau  ≈  0

where div_S is the spherical (tangent-projected) divergence, estimated via
Hutchinson with tangent-projected Rademacher probes, and s_tau is the
score of the frozen flow at z.

We obtain the score via Tweedie + the SFM 'mean' parameterization:
  the model predicts log_p over vocab; x_pred = sum_v p(v) * E[v]
  (the expected clean embedding under the model's beliefs). On the
  sphere the natural "score-like" tangent direction is
      s_tau ~  (x_pred - z * (z . x_pred))            # tangent at z
      i.e. the projection of the predicted-clean displacement onto T_z S.
  This is the analog of  (x_pred - z)/((1-tau)^2 sigma^2)  in Euclidean
  flow-matching, up to a (1-alpha)-dependent positive scalar that is
  absorbed by the penalty's coefficient.

If lambda_div = 0, the loop reduces exactly to variant A.

Headline question (B vs A): does constraining psi toward this subspace
match or beat unconstrained psi on GSM8K accuracy?
"""
import os
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric import Fabric

import utils
import dataloader as dataloader_mod
from samplers import sfm_compute_velocity, SFMContext

from psi.nets.psi_sphere_net import SphereSolenoidalNet, project_tangent
from psi.trainers.psi_train_v1 import (
    _load_frozen_model, _build_prompt_mask,
    _pool_conditioning, _sample_alpha)


def _spherical_score(z, x_pred, eps=1e-8):
    """Tangent-projected 'score-like' direction at z.

    s_tau(z) ~ (I - z z^T) (x_pred - z)
            = (x_pred - z) - z * (z . (x_pred - z))
            = x_pred - z * (z . x_pred)        # using z.z = 1
    """
    # Per-position dot
    dot = (z * x_pred).sum(dim=-1, keepdim=True)
    return x_pred - dot * z


def _hutchinson_tangent_div_and_psi_dot_score(psi_net, z, alpha, c,
                                              score, n_probes=1):
    """Estimate  m_div = div_S(psi)(z) + psi . score   per position, summed.

    Returns:
      m_div_sum_per_b  (B,)    used as (m_div^2).mean() in the loss
      diag dict        for logging norms

    Uses tangent-projected Rademacher probes and one autograd VJP.
    """
    B, L_pos, D = z.shape
    # Per-position estimator: keep the (B, L) shape. Caller squares and
    # means over (B, L), giving a per-token MSE — independent of seq len.
    accum = torch.zeros((B, L_pos), device=z.device, dtype=z.dtype)
    psi_val_for_log = None
    psi_dot_score_for_log = None
    div_for_log = None

    for _ in range(n_probes):
        bern = (torch.randint(0, 2, z.shape,
                              device=z.device, dtype=z.dtype) * 2.0 - 1.0)
        v = project_tangent(bern, z)
        z_req = z.detach().clone().requires_grad_(True)
        psi_val = psi_net(z_req, alpha, c)                 # (B,L,d) tangent
        vp = (v * psi_val).sum()
        g, = torch.autograd.grad(vp, z_req, create_graph=True,
                                 retain_graph=True)
        # Hutchinson sum over d (the position's tangent space), NOT over L.
        div_psi_pp = (g * v).sum(dim=-1)                   # (B, L)
        psi_dot_score_pp = (psi_val * score).sum(dim=-1)   # (B, L)
        m_div_pp = div_psi_pp + psi_dot_score_pp           # (B, L)
        accum = accum + m_div_pp
        with torch.no_grad():
            psi_val_for_log = psi_val.detach()
            div_for_log = div_psi_pp.detach()
            psi_dot_score_for_log = psi_dot_score_pp.detach()

    m_div = accum / max(n_probes, 1)                       # (B, L)
    diag = {
        '|psi|_avg': float(psi_val_for_log.norm(dim=-1).mean().item()),
        '|div_psi|_avg': float(div_for_log.abs().mean().item()),
        '|psi.score|_avg': float(psi_dot_score_for_log.abs().mean().item()),
        '|m_div|_avg': float(m_div.detach().abs().mean().item()),
    }
    return m_div, diag


def _psi_train_v2(diffusion_model, config, logger, tokenizer):
    """Variant B training loop. Called from main.py via mode=psi_train_v2."""
    logger.info('=== Variant B: supervised + tangent-div penalty ===')

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        num_nodes=config.trainer.num_nodes)
    fabric.launch()
    device = fabric.device
    seed = config.seed + fabric.global_rank
    L.seed_everything(seed)

    model = _load_frozen_model(diffusion_model, config, tokenizer, device)
    d = model.backbone.sphere_embed.weight.shape[-1]
    eps = config.algo.eps
    block_size = config.model.length

    psi_hidden = int(getattr(config, 'psi', {}).get('hidden', 512))
    psi_net = SphereSolenoidalNet(d=d, hidden=psi_hidden).to(device)
    # Optional warm-init from a previously-trained psi (e.g. variant A).
    # This lets B start with a useful supervised residual and the penalty
    # only has to ROTATE it toward the div-free subspace, instead of
    # competing with L_A from the zero-init point.
    init_from = getattr(config, 'psi', {}).get('init_from', '')
    if init_from:
        sd = torch.load(init_from, map_location='cpu')
        psi_net.load_state_dict(
            sd['psi_state'] if 'psi_state' in sd else sd)
        logger.info(f'Warm-init psi from {init_from}')
    n_psi = sum(p.numel() for p in psi_net.parameters() if p.requires_grad)
    logger.info(f'psi_net: d={d}, hidden={psi_hidden}, '
                f'#params={n_psi/1e6:.2f}M')

    ds_mode = getattr(config, 'psi', {}).get('dataset_mode', 'tinygsm')
    with fabric.rank_zero_first():
        if ds_mode == 'gsm8k_train':
            from psi.data.psi_dataset import GSM8KTrainDataset
            cache_path = getattr(config, 'psi', {}).get(
                'gsm8k_train_cache',
                '/n/fs/aa-rldiff/winter/s-flm/data/gsm8k_train.json')
            train_ds = GSM8KTrainDataset(
                tokenizer=tokenizer,
                block_size=block_size,
                cache_path=cache_path)
            logger.info(f'GSM8K train: {len(train_ds)} examples '
                        f'(dropped {train_ds.n_dropped} too-long)')
        else:
            train_ds = dataloader_mod.get_dataset(
                config, tokenizer, mode='train')

    def collate(batch):
        ids = torch.stack(
            [torch.as_tensor(b['input_ids'], dtype=torch.long) for b in batch], 0)
        am = torch.stack(
            [torch.as_tensor(b['attention_mask'], dtype=torch.long) for b in batch], 0)
        return ids, am

    loader = DataLoader(
        train_ds, batch_size=config.loader.batch_size,
        shuffle=True, num_workers=config.loader.num_workers,
        collate_fn=collate, drop_last=True)

    lr = float(getattr(config, 'psi', {}).get('lr', 3e-4))
    weight_decay = float(getattr(config, 'psi', {}).get('weight_decay', 0.0))
    optimizer = torch.optim.AdamW(
        psi_net.parameters(), lr=lr, betas=(0.9, 0.95),
        weight_decay=weight_decay)

    n_steps = int(getattr(config, 'psi', {}).get('n_steps', 8000))
    log_every = int(getattr(config, 'psi', {}).get('log_every', 50))
    save_every = int(getattr(config, 'psi', {}).get('save_every', 1000))
    alpha_min = float(getattr(config, 'psi', {}).get('alpha_min', 0.05))
    alpha_max = float(getattr(config, 'psi', {}).get('alpha_max', 0.95))
    lambda_div = float(getattr(config, 'psi', {}).get('lambda_div', 1e-3))
    n_div_probes = int(getattr(config, 'psi', {}).get('n_div_probes', 1))

    out_dir = getattr(config, 'psi', {}).get(
        'out_dir', os.path.join(config.gsm8k.output_dir, 'psi_v2_ckpts'))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'train_log.jsonl')
    logger.info(f'lambda_div = {lambda_div}, n_div_probes = {n_div_probes}')
    logger.info(f'Outputs -> {out_dir}')

    E_full = utils.sphere_normalize(
        model.backbone.sphere_embed.weight.detach())

    psi_net.train()
    loader_iter = iter(loader)
    step = 0
    t0 = time.time()
    running = {'loss': 0.0, 'L_A': 0.0, 'L_div': 0.0}

    while step < n_steps:
        try:
            input_ids, attn_mask = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            input_ids, attn_mask = next(loader_iter)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        B = input_ids.shape[0]

        first_ans = attn_mask.float().argmax(dim=1)
        all_zero = (attn_mask.sum(dim=1) == 0)
        prompt_lens = torch.where(all_zero, torch.zeros_like(first_ans), first_ans)
        prompt_mask = _build_prompt_mask(attn_mask, prompt_lens, block_size)

        # ---- z, v_sfm, v_target, score (no grad on base) ----
        with torch.no_grad():
            x1 = model.backbone.get_sphere_embeddings(input_ids)
            x1 = utils.sphere_normalize(x1).detach()
            x0 = utils.sphere_normalize(torch.randn_like(x1))

            alpha = _sample_alpha(B, alpha_min, alpha_max, device)
            slerp_t = (alpha if model.invert_time_convention
                       else (1.0 - alpha))
            z = model._slerp(x1, x0, slerp_t)
            z = utils.sphere_normalize(z)

            sigma_t = model._sigma_from_alphat(alpha.unsqueeze(-1))
            log_p = model.forward(
                xt=z, sigma=sigma_t,
                context=SFMContext(temperature=1.0))
            log_p = log_p.to(z.dtype)

            E_use = E_full.to(z.dtype)
            v_sfm = sfm_compute_velocity(z, E_use, log_p, mode='exact', eps=eps)
            v_target = utils.log_map(z, x1, eps)

            # Predicted-clean point under the model (used to form the
            # tangent score). x_pred = sum_v p(v) * E_v, then sphere-normalize.
            p = log_p.exp()                               # (B,L,V)
            x_pred = torch.einsum('blv,vd->bld', p, E_use)
            x_pred = utils.sphere_normalize(x_pred)
            score = _spherical_score(z, x_pred)           # tangent at z (B,L,d)

            c = _pool_conditioning(x1, prompt_mask)

        # ---- psi forward (grad ON) for L_A ----
        psi_dtype = next(psi_net.parameters()).dtype
        psi_tan = psi_net(
            z.to(psi_dtype),
            alpha.to(psi_dtype),
            c.to(psi_dtype)).to(z.dtype)
        residual = v_sfm + psi_tan - v_target
        ans_mask = attn_mask.to(z.dtype).unsqueeze(-1)
        L_A = ((residual.pow(2) * ans_mask).sum(dim=-1).sum()
               / ans_mask.sum().clamp(min=1.0))

        # ---- divergence penalty (separate forward via Hutchinson) ----
        # NOTE: this calls psi_net again with grad to support autograd grad.
        z_for_div = z.to(psi_dtype)
        alpha_for_div = alpha.to(psi_dtype)
        c_for_div = c.to(psi_dtype)
        score_for_div = score.to(psi_dtype)
        m_div, div_diag = _hutchinson_tangent_div_and_psi_dot_score(
            psi_net, z_for_div, alpha_for_div, c_for_div,
            score_for_div, n_probes=n_div_probes)
        L_div = (m_div.to(z.dtype) ** 2).mean()

        loss = L_A + lambda_div * L_div

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(psi_net.parameters(), max_norm=1.0)
        optimizer.step()

        running['loss']  += float(loss.item())
        running['L_A']   += float(L_A.item())
        running['L_div'] += float(L_div.item())
        step += 1

        if step % log_every == 0 or step == 1:
            n = min(step, log_every)
            avg = {k: v / n for k, v in running.items()}
            for k in running: running[k] = 0.0
            with torch.no_grad():
                rec = {
                    'step': step,
                    'loss': avg['loss'], 'L_A': avg['L_A'], 'L_div': avg['L_div'],
                    'lambda_div': lambda_div,
                    '|v_sfm|': float(v_sfm.norm(dim=-1).mean().item()),
                    '|v_target|': float(v_target.norm(dim=-1).mean().item()),
                    '|residual|': float(residual.norm(dim=-1).mean().item()),
                    'alpha_mean': float(alpha.mean().item()),
                    'wall_s': time.time() - t0,
                    **div_diag,
                }
            logger.info(json.dumps(rec))
            if fabric.global_rank == 0:
                with open(log_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')

        if step % save_every == 0 and fabric.global_rank == 0:
            ckpt_path = os.path.join(out_dir, f'psi_step{step}.pt')
            torch.save({'psi_state': psi_net.state_dict(),
                        'd': d, 'hidden': psi_hidden,
                        'step': step, 'lambda_div': lambda_div}, ckpt_path)
            torch.save({'psi_state': psi_net.state_dict(),
                        'd': d, 'hidden': psi_hidden,
                        'step': step, 'lambda_div': lambda_div},
                       os.path.join(out_dir, 'psi_latest.pt'))
            logger.info(f'Saved psi (B) checkpoint -> {ckpt_path}')

    if fabric.global_rank == 0:
        torch.save({'psi_state': psi_net.state_dict(),
                    'd': d, 'hidden': psi_hidden,
                    'step': step, 'lambda_div': lambda_div},
                   os.path.join(out_dir, 'psi_final.pt'))
        logger.info('=== ψ training (variant B) done ===')
    fabric.barrier()
