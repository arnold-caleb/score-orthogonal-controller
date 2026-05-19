"""Variant C / B' — REINFORCE on circulation-projected psi.

Uses the proven SFMSamplerWithPsiCirc.step() infrastructure (via run_sampler)
to roll out trajectories with Brownian-tangent noise; records per-step
quantities; computes REINFORCE policy gradient off-line.

Key design choice: the rollout goes through the validated sampler path
that we know reproduces the baseline 12.51% (verified by the sanity job).
The trainer only orchestrates: rollout → verify → recompute psi with grad
→ REINFORCE update.
"""
import os
import json
import math
import time

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric import Fabric

import utils
import dataloader as dataloader_mod
from samplers import run_sampler
import sandbox_gsm8k

from psi.nets.psi_circ_net import CirculationPsiNet
from psi.samplers.psi_circ_sampler import SFMSamplerWithPsiCirc
from psi.trainers.psi_train_v1 import _load_frozen_model as _load_frozen_old


def _load_frozen_model_eval_style(diffusion_model_cls, config, tokenizer, device):
    """Replicate _sample_gsm8k's model-load pattern verbatim (inlined to
    avoid re-importing main and re-registering OmegaConf resolvers)."""
    model = diffusion_model_cls.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer, config=config,
        strict=config.eval.strict_loading)
    model.to(device)
    if config.eval.disable_ema:
        model.ema = None
    model._eval_mode()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─────────────────────────────────────────────────────────────────────
# Verifier (serial — SIGALRM is main-thread only)
# ─────────────────────────────────────────────────────────────────────
def _score_batch(responses, gold_answers, timeout_s=2.0, **_):
    rewards = []
    for resp, gold in zip(responses, gold_answers):
        try:
            ok = sandbox_gsm8k.evaluate_samples(
                resp, gold, timeout_s=timeout_s)
            rewards.append(1.0 if ok else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# ─────────────────────────────────────────────────────────────────────
# REINFORCE loss from a recorded trajectory
# ─────────────────────────────────────────────────────────────────────
def _reinforce_loss(psi_net, trajectory, advantage, sigma_explore):
    """advantage: (B*K,). Returns scalar loss.

    If `trajectory` is empty (e.g. ran with use_internal_sampler=true so
    no trajectory was recorded), there is nothing to learn from — return
    a 0-valued tensor that's autograd-safe.
    """
    if len(trajectory) == 0:
        # Make a dummy 0-loss tied to psi params so .backward() works.
        zero = sum(p.sum() * 0 for p in psi_net.parameters())
        return zero
    log_pi_sum = None
    for entry in trajectory:
        z       = entry['z']
        v_sfm   = entry['v_sfm']
        score   = entry['score']
        c       = entry['c']
        alpha   = entry['alpha']
        dt      = entry['dt']                    # python float
        step_t  = entry['step']

        # Recompute psi with grad
        psi_dtype = next(psi_net.parameters()).dtype
        psi_grad = psi_net(
            z.to(psi_dtype), alpha.to(psi_dtype),
            c.to(psi_dtype), score.to(psi_dtype)).to(z.dtype)
        drift_grad = dt * (v_sfm + psi_grad)

        diff = step_t - drift_grad                # (B*K, gen_len, d)
        sq = (diff * diff).sum(dim=-1)            # (B*K, gen_len)
        # log N(step | drift, sigma² * |dt|) per position
        log_pi_pos = -sq / (2.0 * sigma_explore ** 2 * abs(dt))
        log_pi_b = log_pi_pos.sum(dim=-1)         # (B*K,)
        log_pi_sum = log_pi_b if log_pi_sum is None else log_pi_sum + log_pi_b

    return -(advantage.detach() * log_pi_sum).mean()


def _pad_prefix_batch(token_lists, plens, device, block_size, pad_id):
    """Pad each token sequence to block_size; return (ids, plens) on device."""
    ids = torch.full((len(token_lists), block_size), pad_id, dtype=torch.long)
    for i, t in enumerate(token_lists):
        ids[i, :len(t)] = torch.as_tensor(t, dtype=torch.long)
    return ids.to(device), plens.to(device)


# ─────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────
def _psi_train_c(diffusion_model, config, logger, tokenizer):
    logger.info('=== Variant C: REINFORCE on circulation psi (via SFMSampler) ===')

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        num_nodes=config.trainer.num_nodes)
    fabric.launch()
    device = fabric.device
    seed = config.seed + fabric.global_rank
    L.seed_everything(seed)

    model = _load_frozen_model_eval_style(diffusion_model, config, tokenizer, device)

    # ── EARLY SANITY: replicate the standalone repro_min call EXACTLY ──
    # If this prints Python code, the model is fine right after load.
    # If it prints gibberish, model load is the bug.
    import json as _json
    _test = _json.load(open('/n/fs/aa-rldiff/winter/s-flm/data/gsm8k_test.json'))[0]
    _q = tokenizer(_test['prompt'].strip(), add_special_tokens=False).input_ids
    _sep = tokenizer(config.data.separator, add_special_tokens=False).input_ids
    _ids = [tokenizer.bos_token_id] + _q + _sep
    _ids_t = torch.tensor(_ids, dtype=torch.long, device=device).unsqueeze(0)
    _plens_t = torch.tensor([len(_ids)], device=device)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    with torch.no_grad():
        _toks, _ = model.generate_samples(
            num_samples=1, num_steps=int(config.sampler.steps),
            prefix_tokens=_ids_t, prefix_lengths=_plens_t)
    _out = tokenizer.decode(_toks[0, len(_ids):].cpu().tolist(),
                            skip_special_tokens=True)
    logger.info(f'EARLY-SANITY response: {_out[:300]!r}')
    # ── END EARLY SANITY ──
    d = model.backbone.sphere_embed.weight.shape[-1]
    eps = config.algo.eps
    block_size = config.model.length
    invert_time = bool(model.invert_time_convention)

    psi_hidden = int(getattr(config, 'psi', {}).get('hidden', 512))
    psi_net = CirculationPsiNet(d=d, hidden=psi_hidden).to(device)

    init_from = getattr(config, 'psi', {}).get('init_from', '')
    if init_from:
        sd = torch.load(init_from, map_location='cpu')
        own = psi_net.state_dict()
        in_sd = sd['psi_state'] if 'psi_state' in sd else sd
        matched = {k: v for k, v in in_sd.items()
                   if k in own and own[k].shape == v.shape}
        psi_net.load_state_dict(matched, strict=False)
        logger.info(f'Warm-init psi from {init_from} '
                    f'({len(matched)} matched keys)')

    n_psi = sum(p.numel() for p in psi_net.parameters() if p.requires_grad)
    logger.info(f'CirculationPsiNet: d={d}, hidden={psi_hidden}, '
                f'#params={n_psi/1e6:.2f}M')

    # Dataset
    with fabric.rank_zero_first():
        from psi.data.psi_dataset_rl import GSM8KRLDataset
        cache_path = getattr(config, 'psi', {}).get(
            'gsm8k_train_cache',
            '/n/fs/aa-rldiff/winter/s-flm/data/gsm8k_train.json')
        max_prompt_len = int(getattr(config, 'psi', {}).get(
            'max_prompt_len', block_size // 2))
        train_ds = GSM8KRLDataset(
            tokenizer=tokenizer, block_size=block_size,
            cache_path=cache_path, max_prompt_len=max_prompt_len)
        logger.info(f'GSM8K-RL: {len(train_ds)} prompts '
                    f'(dropped {train_ds.n_dropped} too-long)')

    def collate(batch):
        ids = torch.stack([b['input_ids'] for b in batch], 0)
        plens = torch.tensor([b['prompt_len'] for b in batch], dtype=torch.long)
        golds = [b['gold_answer_str'] for b in batch]
        return ids, plens, golds

    loader = DataLoader(
        train_ds,
        batch_size=int(getattr(config, 'psi', {}).get('prompt_batch', 2)),
        shuffle=True,
        num_workers=int(config.loader.num_workers),
        collate_fn=collate,
        drop_last=True)

    # Hyperparams
    lr        = float(getattr(config, 'psi', {}).get('lr', 1e-4))
    n_steps   = int(getattr(config, 'psi', {}).get('n_steps', 1500))
    K         = int(getattr(config, 'psi', {}).get('K_rollouts', 4))
    n_sample  = int(getattr(config, 'psi', {}).get('sampler_steps', 32))
    sigma_e   = float(getattr(config, 'psi', {}).get('sigma_explore', 0.05))
    log_every = int(getattr(config, 'psi', {}).get('log_every', 10))
    save_every = int(getattr(config, 'psi', {}).get('save_every', 200))
    verifier_timeout = float(getattr(config, 'psi', {}).get(
        'verifier_timeout_s', 2.0))
    grad_clip = float(getattr(config, 'psi', {}).get('grad_clip', 1.0))

    optimizer = torch.optim.AdamW(
        psi_net.parameters(), lr=lr, betas=(0.9, 0.95),
        weight_decay=0.0)

    out_dir = getattr(config, 'psi', {}).get(
        'out_dir', os.path.join(config.gsm8k.output_dir, 'psi_c_ckpts'))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'train_log.jsonl')
    logger.info(f'K={K}, sampler_steps={n_sample}, sigma_explore={sigma_e}, '
                f'lr={lr}, prompt_batch={loader.batch_size}')

    # Pre-build the sampler (will be reused; trajectory reset each rollout)
    sampler = SFMSamplerWithPsiCirc(
        psi_net=psi_net,
        sigma_explore=sigma_e,
        record_trajectory=True,
        noise_removal='greedy',
        velocity='exact',
        use_float64=True,
        slerp_float64=(config.algo.slerp_precision == 'float64'),
        eps=eps, temperature=1.0,
        p_nucleus=1.0, top_k=-1, top_k_velocity=-1,
        invert_time_convention=invert_time)

    psi_net.train()
    loader_iter = iter(loader)
    step = 0
    t0 = time.time()
    running = {'loss': 0.0, 'reward_mean': 0.0, 'reward_max': 0.0,
               'reward_std': 0.0, '|psi|': 0.0}

    while step < n_steps:
        try:
            ids, plens, golds = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            ids, plens, golds = next(loader_iter)
        B = ids.shape[0]
        ids = ids.to(device, non_blocking=True)
        plens = plens.to(device, non_blocking=True)

        # Replicate K times
        ids_rep   = ids.repeat_interleave(K, dim=0)
        plens_rep = plens.repeat_interleave(K)
        golds_rep = [g for g in golds for _ in range(K)]

        # ROLLOUT
        use_internal = bool(getattr(config, 'psi', {}).get(
            'use_internal_sampler', False))
        sampler.reset_trajectory()
        psi_net.eval()

        # LOOP-SANITY (only step 0): re-run Janet's ducks here. If this
        # still produces Python, the prompts/batching are the issue.
        # If it produces gibberish, something in the loop corrupted the model.
        if step == 0 and fabric.global_rank == 0:
            import json as _j
            _t = _j.load(open('/n/fs/aa-rldiff/winter/s-flm/data/gsm8k_test.json'))[0]
            _q = tokenizer(_t['prompt'].strip(), add_special_tokens=False).input_ids
            _sp = tokenizer(config.data.separator, add_special_tokens=False).input_ids
            _id = [tokenizer.bos_token_id] + _q + _sp
            _idt = torch.tensor(_id, dtype=torch.long, device=device).unsqueeze(0)
            _plt = torch.tensor([len(_id)], device=device)
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            with torch.no_grad():
                _tk, _ = model.generate_samples(
                    num_samples=1, num_steps=int(config.sampler.steps),
                    prefix_tokens=_idt, prefix_lengths=_plt)
            _r = tokenizer.decode(_tk[0, len(_id):].cpu().tolist(),
                                  skip_special_tokens=True)
            logger.info(f'LOOP-SANITY response: {_r[:300]!r}')
        # Force a fresh seed BEFORE sampling to match what eval would see if
        # it had been started with the same config.seed (eliminates the RNG
        # offset introduced by psi_net's weight init).
        if step == 0:
            torch.manual_seed(int(config.seed))
            torch.cuda.manual_seed_all(int(config.seed))
        with torch.no_grad():
            if use_internal:
                # Use the model's own SFMSampler (built in __init__) — bypasses
                # SFMSamplerWithPsiCirc entirely. For σ=ψ=0 sanity only.
                tokens, _ = model.generate_samples(
                    num_samples=B * K, num_steps=n_sample, eps=eps,
                    prefix_tokens=ids_rep, prefix_lengths=plens_rep)
            else:
                tokens, _ = run_sampler(
                    sampler, model, num_samples=B * K,
                    num_steps=n_sample, eps=eps,
                    prefix_tokens=ids_rep, prefix_lengths=plens_rep)
        psi_net.train()

        # Decode + verify
        tokens_cpu = tokens.cpu()
        responses = [
            tokenizer.decode(
                tokens_cpu[i, int(plens_rep[i].item()):].tolist(),
                skip_special_tokens=True)
            for i in range(B * K)]
        rewards = _score_batch(
            responses, golds_rep, timeout_s=verifier_timeout)
        R = torch.tensor(rewards, dtype=torch.float32, device=device)

        # First-step sample dump
        if step == 0 and fabric.global_rank == 0:
            logger.info('--- First rollout sample dump ---')
            for i in range(min(2, B * K)):
                prompt_str = tokenizer.decode(
                    ids_rep[i, :int(plens_rep[i].item())].tolist(),
                    skip_special_tokens=True)
                logger.info(f'[sample {i}] reward={rewards[i]}')
                logger.info(f'  prompt: {prompt_str[:160]}')
                logger.info(f'  response: {responses[i][:300]}')
                logger.info(f'  gold: {golds_rep[i][:120]}')
            logger.info('--- end dump ---')

        # Per-prompt-mean baseline
        R_bk = R.view(B, K)
        baseline = R_bk.mean(dim=-1, keepdim=True).expand(-1, K).reshape(-1)
        advantage = R - baseline

        # Skip if all rewards in a batch identical → no signal
        if R_bk.std(dim=-1).sum().item() < 1e-8:
            zero_var_skip = True
            loss_val = 0.0
        else:
            zero_var_skip = False
            loss = _reinforce_loss(
                psi_net, sampler.trajectory, advantage, sigma_e)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(psi_net.parameters(),
                                           max_norm=grad_clip)
            optimizer.step()
            loss_val = float(loss.item())

        # Diagnostics
        with torch.no_grad():
            psi_norms = [float(e['psi'].norm(dim=-1).mean().item())
                         for e in sampler.trajectory
                         if e['psi'] is not None]
            psi_norm_avg = (sum(psi_norms) / len(psi_norms)) if psi_norms else 0.0

        running['loss']        += loss_val
        running['reward_mean'] += float(R.mean().item())
        running['reward_max']  += float(R.max().item())
        running['reward_std']  += float(R.std().item()) if R.numel() > 1 else 0.0
        running['|psi|']       += psi_norm_avg
        step += 1

        if step % log_every == 0 or step == 1:
            n = min(step, log_every)
            avg = {k: v / n for k, v in running.items()}
            for k in running: running[k] = 0.0
            rec = {
                'step': step, 'loss': avg['loss'],
                'reward_mean': avg['reward_mean'],
                'reward_max': avg['reward_max'],
                'reward_std': avg['reward_std'],
                '|psi|': avg['|psi|'],
                'sigma_explore': sigma_e, 'K': K, 'B': B,
                'zero_var_skip_this_step': zero_var_skip,
                'wall_s': time.time() - t0,
            }
            logger.info(json.dumps(rec))
            if fabric.global_rank == 0:
                with open(log_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')

        if step % save_every == 0 and fabric.global_rank == 0:
            ckpt = {'psi_state': psi_net.state_dict(),
                    'd': d, 'hidden': psi_hidden, 'step': step,
                    'sigma_explore': sigma_e, 'K': K,
                    'sampler_steps': n_sample}
            torch.save(ckpt, os.path.join(out_dir, f'psi_step{step}.pt'))
            torch.save(ckpt, os.path.join(out_dir, 'psi_latest.pt'))
            logger.info(f'Saved psi (C) ckpt -> psi_step{step}.pt')

    if fabric.global_rank == 0:
        torch.save({'psi_state': psi_net.state_dict(),
                    'd': d, 'hidden': psi_hidden, 'step': step,
                    'sigma_explore': sigma_e, 'K': K,
                    'sampler_steps': n_sample},
                   os.path.join(out_dir, 'psi_final.pt'))
        logger.info('=== Variant C training done ===')
    fabric.barrier()
