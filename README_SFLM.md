# Language Modeling with Hyperspherical Flows

By [Justin Deschenaux](https://jdeschena.com) and [Caglar Gulcehre](https://www.caglar.ai).

[![arXiv](https://img.shields.io/badge/arXiv-2605.11125-red.svg)](https://arxiv.org/abs/2605.11125)
[![Blog](https://img.shields.io/badge/Blog%20%20-8A2BE2)](https://jdeschena.com/blog/sfm)
[![HuggingFace](https://img.shields.io/badge/🤗-Huggingface-blue)](https://huggingface.co/jdeschena/s-flm)


**Abstract**: Continuous Flow Language Models (FLMs) transport noise to data with a deterministic ODE, avoiding the factorized-sampling assumption of discrete diffusion. But standard FLMs operate on one-hot vectors whose dimension scales with the vocabulary, making them expensive to train. 𝕊-FLM instead operate on the hypersphere, where we transport random points towards the clean embeddings via rotations. Previous FLMs match AR in Generative Perplexity, **but high-likelihood samples are not necessarily correct in verifiable domains like math and code**; 𝕊-FLM substantially improves over previous FLMs on GSM8K!

This repository contains training and evaluation code for 𝕊-FLM along with the discrete-diffusion / flow-matching baselines we compare against (AR, MDLM, Duo, FLM, CANDI). We release pretrained checkpoints for two settings:

- **TinyGSM**: math reasoning, SmolLM-135M tokenizer, 250k steps.
- **OpenWebText (OWT)**: general LM, GPT-2 tokenizer, 1M steps.

In addition, we provide training and evaluation scripts (no released checkpoints) for **Sudoku** (synthetic puzzle dataset, 9-digit vocab + clue-tokens, 20k steps; difficulty levels: easy / medium / hard).

[Getting started](#getting-started) · [Checkpoints](#checkpoints) · [Training](#training) · [Sampling & evaluation](#sampling--evaluation) · [Citation](#citation)

# Getting started

Create a fresh environment and install the Python dependencies:

```bash
conda create -n sfm python=3.12
conda activate sfm
pip install -r requirements.txt
```

`requirements.txt` intentionally does **not** pin `torch` or `numpy`. We work inside the NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.02-py3`) which already ships matching CUDA / cuDNN / NCCL builds; pip-installing torch on top of that will mess up the NGC build. If you are not using the container, install `torch` and `numpy` (we use `torch==2.7.0`, `numpy==1.26.4`) **before** running `pip install -r requirements.txt`.

# Checkpoints

All the checkpoints are in the HuggingFace repo [`jdeschena/s-flm`](https://huggingface.co/jdeschena/s-flm).

### Layout on Huggingface

```
tinygsm/                                              # trained on TinyGSM, 250k steps
  ar.ckpt
  mdlm.ckpt
  duo.ckpt
  candi/{lr3e-4,lr1e-3}.ckpt                          # CANDI uses a smaller learning rate in certain configs, hence we try with both
  flm/{default,caps}.ckpt                             # FLMs use attention softcapping and a custom logits processing. We experiment with and without.
  sfm/sphere_dit_truncated_fixed_no_renorm.ckpt       # Standard DiT, truncated schedule
  sfm/sphere_dit_truncated_adaptive_no_renorm.ckpt    # Standard DiT, truncated+adaptive schedule
  sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt   # S-arch (nGPT-inspired), truncated+adaptive schedule

owt/                                                  # trained on OpenWebText, 1M steps
  ar.ckpt
  mdlm.ckpt
  duo.ckpt
  flm.ckpt                                            # Original FLM checkpoint of https://github.com/david3684/flm
  sfm.ckpt                                            # S-arch, truncated+adaptive schedule
```

### Download

A single checkpoint:

```bash
huggingface-cli download jdeschena/s-flm tinygsm/duo.ckpt \
    --local-dir ./checkpoints
```

The whole repo (or a subset):

```bash
# Everything (~47 GB):
huggingface-cli download jdeschena/s-flm --local-dir ./checkpoints

# Just TinyGSM:
huggingface-cli download jdeschena/s-flm --local-dir ./checkpoints \
    --include 'tinygsm/**'
```

# Training

All training scripts are in `scripts/train/`. Each script is self-contained and exposes a few environment variables for overrides: `OUTPUT_DIR`, `CACHE_DIR`, `NUM_NODES`, `DEVICES`. They were originally run on 2 nodes with 4 GPUs (TinyGSM) or 4 nodes with 4 GPUs (OWT).

| Dataset | Algorithm | Script |
|---|---|---|
| TinyGSM | AR | [`scripts/train/tinygsm/ar.sh`](scripts/train/tinygsm/ar.sh) |
| TinyGSM | MDLM | [`scripts/train/tinygsm/mdlm.sh`](scripts/train/tinygsm/mdlm.sh) |
| TinyGSM | Duo | [`scripts/train/tinygsm/duo.sh`](scripts/train/tinygsm/duo.sh) |
| TinyGSM | FLM (default) | [`scripts/train/tinygsm/flm_default.sh`](scripts/train/tinygsm/flm_default.sh) |
| TinyGSM | FLM (w/ softcapping) | [`scripts/train/tinygsm/flm_caps.sh`](scripts/train/tinygsm/flm_caps.sh) |
| TinyGSM | CANDI (lr 3e-4) | [`scripts/train/tinygsm/candi_lr3e-4.sh`](scripts/train/tinygsm/candi_lr3e-4.sh) |
| TinyGSM | CANDI (lr 1e-3) | [`scripts/train/tinygsm/candi_lr1e-3.sh`](scripts/train/tinygsm/candi_lr1e-3.sh) |
| TinyGSM | SFM: sphere-DiT, fixed truncated | [`scripts/train/tinygsm/sfm_sphere_dit_truncated_fixed_no_renorm.sh`](scripts/train/tinygsm/sfm_sphere_dit_truncated_fixed_no_renorm.sh) |
| TinyGSM | SFM: sphere-DiT, adaptive truncated | [`scripts/train/tinygsm/sfm_sphere_dit_truncated_adaptive_no_renorm.sh`](scripts/train/tinygsm/sfm_sphere_dit_truncated_adaptive_no_renorm.sh) |
| TinyGSM | SFM: SphereArch, adaptive truncated | [`scripts/train/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh`](scripts/train/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh) |
| OWT | AR | [`scripts/train/owt/ar.sh`](scripts/train/owt/ar.sh) |
| OWT | MDLM | [`scripts/train/owt/mdlm.sh`](scripts/train/owt/mdlm.sh) |
| OWT | Duo | [`scripts/train/owt/duo.sh`](scripts/train/owt/duo.sh) |
| OWT | FLM | [`scripts/train/owt/flm.sh`](scripts/train/owt/flm.sh) |
| OWT | SFM | [`scripts/train/owt/sfm.sh`](scripts/train/owt/sfm.sh) |
| Sudoku | AR | [`scripts/train/sudoku/ar.sh`](scripts/train/sudoku/ar.sh) |
| Sudoku | MDLM | [`scripts/train/sudoku/mdlm.sh`](scripts/train/sudoku/mdlm.sh) |
| Sudoku | Duo | [`scripts/train/sudoku/duo.sh`](scripts/train/sudoku/duo.sh) |
| Sudoku | FLM | [`scripts/train/sudoku/flm.sh`](scripts/train/sudoku/flm.sh) |
| Sudoku | CANDI | [`scripts/train/sudoku/candi.sh`](scripts/train/sudoku/candi.sh) |
| Sudoku | SFM | [`scripts/train/sudoku/sfm.sh`](scripts/train/sudoku/sfm.sh) |
| Sudoku | SFM (truncated) | [`scripts/train/sudoku/sfm_truncated.sh`](scripts/train/sudoku/sfm_truncated.sh) |
| Sudoku | SFM (truncated + adaptive) | [`scripts/train/sudoku/sfm_truncated_adaptive.sh`](scripts/train/sudoku/sfm_truncated_adaptive.sh) |

Sudoku scripts accept a `DIFFICULTY` environment variable (`easy` / `medium` / `hard`).

Example:

```bash
DEVICES=8 NUM_NODES=1 \
OUTPUT_DIR=./outputs/sfm_tinygsm \
bash scripts/train/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh
```

# Sampling & evaluation

Sampling/eval scripts are in `scripts/sample/` and expose the same overrides plus `CKPT_PATH` and `STEPS` (default 32).

For TinyGSM, the sampling script uses the GSM8K test and writes a per-example JSON to `OUTPUT_DIR`; On OWT, we generate unconditional samples and computes generative perplexity.

| Setting | Algorithm | Script |
|---|---|---|
| TinyGSM | AR | [`scripts/sample/tinygsm/ar.sh`](scripts/sample/tinygsm/ar.sh) |
| TinyGSM | MDLM | [`scripts/sample/tinygsm/mdlm.sh`](scripts/sample/tinygsm/mdlm.sh) |
| TinyGSM | Duo | [`scripts/sample/tinygsm/duo.sh`](scripts/sample/tinygsm/duo.sh) |
| TinyGSM | FLM (default) | [`scripts/sample/tinygsm/flm_default.sh`](scripts/sample/tinygsm/flm_default.sh) |
| TinyGSM | FLM (w/ softcapping) | [`scripts/sample/tinygsm/flm_caps.sh`](scripts/sample/tinygsm/flm_caps.sh) |
| TinyGSM | CANDI (lr 3e-4) | [`scripts/sample/tinygsm/candi_lr3e-4.sh`](scripts/sample/tinygsm/candi_lr3e-4.sh) |
| TinyGSM | CANDI (lr 1e-3) | [`scripts/sample/tinygsm/candi_lr1e-3.sh`](scripts/sample/tinygsm/candi_lr1e-3.sh) |
| TinyGSM | SFM: sphere-DiT, fixed truncated | [`scripts/sample/tinygsm/sfm_sphere_dit_truncated_fixed_no_renorm.sh`](scripts/sample/tinygsm/sfm_sphere_dit_truncated_fixed_no_renorm.sh) |
| TinyGSM | SFM: sphere-DiT, adaptive truncated | [`scripts/sample/tinygsm/sfm_sphere_dit_truncated_adaptive_no_renorm.sh`](scripts/sample/tinygsm/sfm_sphere_dit_truncated_adaptive_no_renorm.sh) |
| TinyGSM | SFM: SphereArch, adaptive truncated | [`scripts/sample/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh`](scripts/sample/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh) |
| OWT | AR | [`scripts/sample/owt/ar.sh`](scripts/sample/owt/ar.sh) |
| OWT | MDLM | [`scripts/sample/owt/mdlm.sh`](scripts/sample/owt/mdlm.sh) |
| OWT | Duo | [`scripts/sample/owt/duo.sh`](scripts/sample/owt/duo.sh) |
| Sudoku | AR | [`scripts/sample/sudoku/ar.sh`](scripts/sample/sudoku/ar.sh) |
| Sudoku | MDLM | [`scripts/sample/sudoku/mdlm.sh`](scripts/sample/sudoku/mdlm.sh) |
| Sudoku | Duo | [`scripts/sample/sudoku/duo.sh`](scripts/sample/sudoku/duo.sh) |
| Sudoku | FLM | [`scripts/sample/sudoku/flm.sh`](scripts/sample/sudoku/flm.sh) |
| Sudoku | CANDI | [`scripts/sample/sudoku/candi.sh`](scripts/sample/sudoku/candi.sh) |
| Sudoku | SFM | [`scripts/sample/sudoku/sfm.sh`](scripts/sample/sudoku/sfm.sh) |
| Sudoku | SFM (truncated) | [`scripts/sample/sudoku/sfm_truncated.sh`](scripts/sample/sudoku/sfm_truncated.sh) |
| Sudoku | SFM (truncated + adaptive) | [`scripts/sample/sudoku/sfm_truncated_adaptive.sh`](scripts/sample/sudoku/sfm_truncated_adaptive.sh) |
| OWT | FLM | [`scripts/sample/owt/flm.sh`](scripts/sample/owt/flm.sh) |
| OWT | SFM | [`scripts/sample/owt/sfm.sh`](scripts/sample/owt/sfm.sh) |

Example: evaluate an SFM checkpoint on TinyGSM after 64 sampling steps:

```bash
CKPT_PATH=./checkpoints/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt \
STEPS=64 \
OUTPUT_DIR=./eval_runs/sfm_tinygsm \
bash scripts/sample/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh
```

### SFM sampling modes

The SFM sampling scripts (all 3 TinyGSM SFM scripts and `scripts/sample/owt/sfm.sh`) expose two extra environment variables:

- `VELOCITY`: `exact` (default, deterministic velocity) or `sample` (sample from the velocity distribution).
- `TOPK_VELOCITY`: `-1` (default, full vocab — no top-k filtering) or a positive integer (e.g. `1`, `10`) to restrict the velocity to the top-k tokens.

```bash
# Default: exact velocity, full vocab (no top-k).
bash scripts/sample/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh

# Top-1 velocity, sampled (paper's headline setup).
TOPK_VELOCITY=1 VELOCITY=sample bash scripts/sample/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh

# Top-10 velocity, exact.
TOPK_VELOCITY=10 bash scripts/sample/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh

# No top-k (full vocab), sampled velocity.
VELOCITY=sample bash scripts/sample/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh
```

# Acknowledgements

This codebase builds on a number of excellent open-source projects:

- [**Duo**](https://github.com/s-sahoo/duo): discrete diffusion baseline; our overall training/eval scaffolding is descended from theirs.
- [**FLM**](https://github.com/david3684/flm): original Flow Language Model implementation; our OWT FLM checkpoint is the released checkpoint from this repo, and our FLM training/sampling code follows it.
- [**CANDI**](https://github.com/patrickpynadath1/candi-diffusion): continuous-and-discrete diffusion baseline (imported).
- [**PUMA**](https://github.com/JaeyeonKim01/PUMA): reference for the TinyGSM data preparation.
- [**PRISM**](https://github.com/JaeyeonKim01/PRISM): reference for the Sudoku data preparation.

# Citation

```
@misc{deschenaux2026languagemodelinghypersphericalflows,
      title={Language Modeling with Hyperspherical Flows}, 
      author={Justin Deschenaux and Caglar Gulcehre},
      year={2026},
      eprint={2605.11125},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.11125}, 
}
```
