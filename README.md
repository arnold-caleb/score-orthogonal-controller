# score-orthogonal-controller

Experiments on **frozen flow control via tangent residuals** on a continuous-time
flow language model (S-FLM). The base S-FLM repo is preserved as-is (see
[`README_SFLM.md`](README_SFLM.md)); our additions live under `psi/`.

We ask:

> Can a small learned residual vector field `ψ(z, τ, c)`, added to a frozen flow
> LM's velocity during sampling, improve verifiable reasoning accuracy (GSM8K)
> **without** updating the base model — and does constraining `ψ` to live in
> the score-orthogonal subspace of the frozen flow's velocity field (a
> level-set / "circulation-like" controller) buy us anything over an
> unconstrained residual?

The base model is the public S-FLM TinyGSM sphere-arch checkpoint
(135M params; GSM8K baseline 12.51% at 32 sampler steps). We never touch its
weights — `ψ` is a separate ~1.86M-param FiLM-MLP whose output is
**hard-projected** to the sphere tangent at `z` (always) and optionally
also to the orthogonal complement of the spherical score (the
score-orthogonal / circulation variant).

## Variants

| Trainer | Objective | Constraint on ψ | Status |
|---|---|---|---|
| **A** `psi.trainers.v1_supervised`  | Match slerp-velocity target via MSE             | tangent-to-sphere only | A = 9.86% (hurt baseline) |
| **B** `psi.trainers.v2_divfree`     | A + soft `div_S(ψ) + ψ·s` penalty                | tangent + soft div penalty | B = 12.51% (penalty pinned ψ→0; null) |
| **C** `psi.trainers.v3_reinforce`   | REINFORCE on GSM8K verifier reward               | **hard score-orthogonal** | active debugging |

> ⚠️ The `CirculationPsiNet` name is for code continuity only.
> The hard projection enforces `ψ ⊥ z` **and** `ψ ⊥ ŝ_τ`, which is a
> **score-orthogonal / level-set tangent controller** — *not* exact
> `∇·(p_τ ψ) = 0` (would also require `∇_S·ψ = 0`).

## Layout

```
psi/
├── nets/             # ψ architectures (sphere-tangent, score-orthogonal)
├── samplers/         # SFM samplers with ψ injection + trajectory recording
├── trainers/         # variant A / B / C trainers
├── data/             # GSM8K loaders (train-target and RL-prompt formats)
├── diagnostics/      # standalone reproducers & comparison scripts
└── slurms/           # SLURM submit scripts
```

The base S-FLM repo is the rest of this tree (left intact, with small additive
edits to `main.py` and `samplers.py` to dispatch to our trainers and samplers).

## How to run

```bash
# Baseline sanity (verify checkpoint reproduces 12.51%)
sbatch psi/slurms/baseline.slurm
sbatch psi/slurms/sanity_zero.slurm   # sfm_psi with ψ=0 → should also be 12.51%

# Train + eval a variant
sbatch psi/slurms/train_v1.slurm   # variant A: supervised
sbatch psi/slurms/train_v2.slurm   # variant B: div penalty
sbatch psi/slurms/train_v3.slurm   # variant C: REINFORCE

# Eval a trained ψ on GSM8K test
PSI_CKPT=outputs/psi_v1A/psi_step8000.pt TAG=v1A sbatch psi/slurms/eval.slurm
```

## Gotchas (load-bearing)

- The frozen model uses EMA weights. The trainer must call `model._eval_mode()`
  AFTER load to swap EMA shadow params into the main parameters. Setting
  `eval.disable_ema=true` nulls `model.ema` BEFORE `_eval_mode()` runs, which
  silently leaves the model on the raw (non-EMA) weights → produces gibberish.
- The base S-FLM model is in EVAL mode at sample time (via `backbone.eval()`
  inside `_eval_mode()`); do not put it back in train mode.
- `flash-attn` 2.8+ uses `torch.library.wrap_triton` (added in torch 2.5);
  we monkey-patch it to identity at the top of `main.py` for torch 2.4.
- The YAML `separator: '\n'` is the **literal** two-character `\n` (backslash
  + n), not a newline. The frozen model was trained with this literal
  separator.

## Results so far

- Baseline (no ψ, 32 steps): **12.51%** (165/1319)
- Sanity (sfm_psi sampler with ψ=0): **12.51%** ✓ identical
- Variant A (\|ψ\| ≈ 0.5, unconstrained tangent residual): **9.86%** — *hurt*
  (off-manifold failure mode)
- A scaled at inference: ≈ baseline at \|ψ\| ≲ 0.25, ≈ 9.86% at \|ψ\| ≈ 0.5
- Variant B (soft div penalty): \|ψ\| collapsed to ≈ 0.001; **= baseline**
  (the penalty conflicted with the supervised target → ψ→0, never visited
  the constrained subspace)
- Variant C (hard projection + REINFORCE): active debugging
