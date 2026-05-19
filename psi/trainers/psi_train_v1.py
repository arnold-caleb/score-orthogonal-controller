"""Variant A — supervised target matching for frozen-flow tangent residuals.

Trains a small SphereSolenoidalNet psi(z, alpha, c) so that on training pairs
(prompt q, correct answer a):

    L_A(phi) = E_{(q,a), x0, alpha} [
                || M . ( v_sfm(z) + psi(z, alpha, c) - v_target(z) ) ||^2
              ]

where
  x1 = sphere embeddings of (q + a)
  x0 = uniform sphere noise
  z  = slerp(x1, x0, slerp_t)        # SFM convention: slerp_t = 1 - alpha
  v_sfm = sfm_compute_velocity(z, E, log_p) under the FROZEN model
  v_target = log_map(z, x1)           # SFM 'sample'-mode velocity if target=x1
  c = mean over prompt positions of x1
  M = answer-position mask

Base model is frozen; only psi parameters are updated.

This is run as `python -m main mode=psi_train_v1 ...` via a dispatch hook in
main.py. Variant B will subclass this loop and add a divergence penalty.
"""
import os
import json
import math
import time
import contextlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric import Fabric

import utils
import dataloader as dataloader_mod
from samplers import sfm_compute_velocity, SFMContext

from psi.nets.psi_sphere_net import SphereSolenoidalNet


def _load_frozen_model(diffusion_model_cls, config, tokenizer, device):
    """Build the SFM Lightning module from checkpoint, freeze, eval."""
    ckpt = config.eval.checkpoint_path
    assert ckpt, 'set eval.checkpoint_path to the S-FLM ckpt'
    model = diffusion_model_cls.load_from_checkpoint(
        ckpt, config=config, tokenizer=tokenizer,
        strict=config.eval.strict_loading)
    model = model.to(device)
    if hasattr(model, 'ema') and model.ema is not None:
        # mirror gsm8k_eval behavior
        if getattr(config.eval, 'disable_ema', False):
            model.ema = None
    # CRITICAL: swap EMA weights into the main model — this is what
    # _sample_gsm8k does. Without it, model.forward uses raw checkpoint
    # weights and produces gibberish.
    if hasattr(model, '_eval_mode'):
        model._eval_mode()
    # NOTE: _sample_gsm8k does NOT call model.eval(). Some Lightning
    # modules behave differently in train vs eval mode (dropout, etc.).
    # To match eval exactly we leave the module in whatever mode
    # load_from_checkpoint puts it (which is train mode).
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _build_prompt_mask(attention_mask, prompt_lens, block_size):
    """attention_mask is the answer-loss mask (1 on answer tokens).
    We also want the prompt mask (1 on prompt tokens) for pooling
    conditioning c. Build it from prompt_lens.
    """
    B = attention_mask.shape[0]
    positions = torch.arange(block_size, device=attention_mask.device)
    prompt_mask = positions[None, :] < prompt_lens[:, None]   # (B, L)
    return prompt_mask


def _pool_conditioning(x1, prompt_mask):
    """c = mean over prompt positions of x1.  (B, d)."""
    pm = prompt_mask.to(x1.dtype).unsqueeze(-1)               # (B, L, 1)
    c = (x1 * pm).sum(dim=1) / pm.sum(dim=1).clamp(min=1.0)
    return c


def _sample_alpha(B, alpha_min, alpha_max, device):
    """Uniform alpha in [alpha_min, alpha_max].  (B,)."""
    u = torch.rand(B, device=device)
    return alpha_min + (alpha_max - alpha_min) * u


def _psi_train_v1(diffusion_model, config, logger, tokenizer):
    """Variant A training loop. Called from main.py via mode=psi_train_v1."""
    logger.info('=== Variant A: supervised target-matching ψ training ===')

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        num_nodes=config.trainer.num_nodes)
    fabric.launch()
    device = fabric.device
    seed = config.seed + fabric.global_rank
    L.seed_everything(seed)

    # ---- frozen base model ----
    model = _load_frozen_model(diffusion_model, config, tokenizer, device)
    d = model.backbone.sphere_embed.weight.shape[-1]
    eps = config.algo.eps
    block_size = config.model.length
    logger.info(f'Frozen model loaded. d={d}, block_size={block_size}, '
                f'#frozen_params={sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    # ---- psi net ----
    psi_hidden = int(getattr(config, 'psi', {}).get('hidden', 512))
    psi_net = SphereSolenoidalNet(d=d, hidden=psi_hidden).to(device)
    n_psi = sum(p.numel() for p in psi_net.parameters() if p.requires_grad)
    logger.info(f'psi_net: d={d}, hidden={psi_hidden}, #params={n_psi/1e6:.2f}M')

    # ---- dataloader ----
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
        train_ds,
        batch_size=config.loader.batch_size,
        shuffle=True,
        num_workers=config.loader.num_workers,
        collate_fn=collate,
        drop_last=True)

    # ---- optimizer ----
    lr = float(getattr(config, 'psi', {}).get('lr', 3e-4))
    weight_decay = float(getattr(config, 'psi', {}).get('weight_decay', 0.0))
    optimizer = torch.optim.AdamW(
        psi_net.parameters(), lr=lr, betas=(0.9, 0.95),
        weight_decay=weight_decay)

    n_steps = int(getattr(config, 'psi', {}).get('n_steps', 5000))
    log_every = int(getattr(config, 'psi', {}).get('log_every', 50))
    save_every = int(getattr(config, 'psi', {}).get('save_every', 1000))
    alpha_min = float(getattr(config, 'psi', {}).get('alpha_min', 0.05))
    alpha_max = float(getattr(config, 'psi', {}).get('alpha_max', 0.95))

    out_dir = getattr(config, 'psi', {}).get(
        'out_dir', os.path.join(config.gsm8k.output_dir, 'psi_ckpts'))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'train_log.jsonl')
    logger.info(f'Outputs -> {out_dir}')

    # ---- training loop ----
    # Pre-fetch sphere codebook once; it's frozen.
    E_full = utils.sphere_normalize(
        model.backbone.sphere_embed.weight.detach())          # (V, d) float32
    if getattr(config.algo, 'slerp_precision', 'float64') == 'float64':
        E_full64 = E_full.to(torch.float64)
    else:
        E_full64 = None

    psi_net.train()
    loader_iter = iter(loader)
    step = 0
    t0 = time.time()
    running_loss = 0.0

    while step < n_steps:
        try:
            input_ids, attn_mask = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            input_ids, attn_mask = next(loader_iter)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        B = input_ids.shape[0]

        # prompt_lens: we don't carry prompt_len through the wrapped dataset,
        # so derive it as "first index where attention_mask flips from 0 to 1".
        # (For tiny_gsm with train_on_prompt=False and wrap=False, attn_mask
        #  is 1 on answer positions and 0 elsewhere.)
        # First answer position = argmax of attn_mask along L:
        first_ans = attn_mask.float().argmax(dim=1)            # (B,)
        # If attn_mask is all zeros (degenerate), set prompt_len = 0.
        all_zero = (attn_mask.sum(dim=1) == 0)
        prompt_lens = torch.where(all_zero, torch.zeros_like(first_ans),
                                  first_ans)
        prompt_mask = _build_prompt_mask(attn_mask, prompt_lens, block_size)

        # ---- build x1, x0, z = slerp(x1, x0, slerp_t) ----
        with torch.no_grad():
            x1 = model.backbone.get_sphere_embeddings(input_ids)   # (B,L,d)
            x1 = utils.sphere_normalize(x1).detach()
            x0 = utils.sphere_normalize(torch.randn_like(x1))

            alpha = _sample_alpha(B, alpha_min, alpha_max, device)  # (B,)
            slerp_t = (alpha if model.invert_time_convention
                       else (1.0 - alpha))                          # (B,)

            # Use the algo's _slerp (handles float64 promotion + eps)
            z = model._slerp(x1, x0, slerp_t)                       # (B,L,d)
            z = utils.sphere_normalize(z)

            # ---- frozen velocity v_sfm ----
            sigma_t = model._sigma_from_alphat(alpha.unsqueeze(-1))  # (B,1)
            log_p = model.forward(
                xt=z, sigma=sigma_t,
                context=SFMContext(temperature=1.0))                 # (B,L,V)
            log_p = log_p.to(z.dtype)

            E_use = E_full64 if (E_full64 is not None
                                 and z.dtype == torch.float64) else E_full
            E_use = E_use.to(z.dtype)
            v_sfm = sfm_compute_velocity(
                z, E_use, log_p, mode='exact', eps=eps)              # (B,L,d)
            v_target = utils.log_map(z, x1, eps)                     # (B,L,d)

            # Conditioning c (prompt mean of x1)
            c = _pool_conditioning(x1, prompt_mask)                  # (B,d)

        # ---- psi forward (grad ON for psi only) ----
        psi_dtype = next(psi_net.parameters()).dtype
        psi_tan = psi_net(
            z.to(psi_dtype),
            alpha.to(psi_dtype),
            c.to(psi_dtype))                                         # (B,L,d)
        psi_tan = psi_tan.to(z.dtype)

        residual = v_sfm + psi_tan - v_target                        # (B,L,d)

        # Loss only on answer positions
        ans_mask = attn_mask.to(z.dtype).unsqueeze(-1)               # (B,L,1)
        sq = (residual.pow(2) * ans_mask).sum(dim=-1)                # (B,L)
        denom = ans_mask.sum().clamp(min=1.0)
        loss = sq.sum() / denom

        # ---- step ----
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # gentle clip
        torch.nn.utils.clip_grad_norm_(psi_net.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += float(loss.item())
        step += 1

        if step % log_every == 0 or step == 1:
            avg = running_loss / max(1, min(step, log_every))
            running_loss = 0.0
            # also report norms for sanity
            with torch.no_grad():
                vsfm_n = float(v_sfm.norm(dim=-1).mean().item())
                vtgt_n = float(v_target.norm(dim=-1).mean().item())
                psi_n  = float(psi_tan.norm(dim=-1).mean().item())
                resid_n= float(residual.norm(dim=-1).mean().item())
            rec = {'step': step, 'loss': avg,
                   '|v_sfm|': vsfm_n, '|v_target|': vtgt_n,
                   '|psi|': psi_n, '|residual|': resid_n,
                   'alpha_mean': float(alpha.mean().item()),
                   'wall_s': time.time() - t0}
            logger.info(json.dumps(rec))
            if fabric.global_rank == 0:
                with open(log_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')

        if step % save_every == 0 and fabric.global_rank == 0:
            ckpt_path = os.path.join(out_dir, f'psi_step{step}.pt')
            torch.save({'psi_state': psi_net.state_dict(),
                        'config_psi': dict(getattr(config, 'psi', {})),
                        'd': d, 'hidden': psi_hidden, 'step': step},
                       ckpt_path)
            torch.save({'psi_state': psi_net.state_dict(),
                        'config_psi': dict(getattr(config, 'psi', {})),
                        'd': d, 'hidden': psi_hidden, 'step': step},
                       os.path.join(out_dir, 'psi_latest.pt'))
            logger.info(f'Saved psi checkpoint -> {ckpt_path}')

    if fabric.global_rank == 0:
        torch.save({'psi_state': psi_net.state_dict(),
                    'config_psi': dict(getattr(config, 'psi', {})),
                    'd': d, 'hidden': psi_hidden, 'step': step},
                   os.path.join(out_dir, 'psi_final.pt'))
        logger.info('=== ψ training (variant A) done ===')
    fabric.barrier()
