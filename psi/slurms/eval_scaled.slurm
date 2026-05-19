#!/bin/bash
#SBATCH --job-name=sflm_eval_scl
#SBATCH --output=/n/fs/aa-rldiff/winter/s-flm/outputs/eval_psi_scaled_%j.out
#SBATCH --error=/n/fs/aa-rldiff/winter/s-flm/outputs/eval_psi_scaled_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:l40:1

# Magnitude-control eval: scale a trained psi at inference. Lets us test
# whether B's effect (small + div-free psi) is matched by "small A-psi"
# at the same norm — which would mean magnitude is the explanatory
# variable, not the divergence-free direction.
#
# Required env vars:
#   PSI_CKPT  - path to .pt (we'll re-use A's ckpt for the control)
#   PSI_SCALE - multiplicative scale applied to psi at inference
#   TAG       - short label (e.g. 'A_scaled0p07')

set -uo pipefail

: "${PSI_CKPT:?Set PSI_CKPT}"
: "${PSI_SCALE:?Set PSI_SCALE}"
: "${TAG:?Set TAG}"

export TMPDIR=/n/fs/aa-rldiff/tmp
export XDG_CACHE_HOME=/n/fs/aa-rldiff/.cache
export HF_HOME=/n/fs/aa-rldiff/hf_cache_global
export TRANSFORMERS_CACHE=/n/fs/aa-rldiff/hf_cache_global
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

export SFLM_PSI_D=768
export SFLM_PSI_HIDDEN=512
export SFLM_PSI_CKPT="${PSI_CKPT}"
export SFLM_PSI_SCALE="${PSI_SCALE}"

source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda/envs/drifting

mkdir -p /n/fs/aa-rldiff/winter/s-flm/outputs
mkdir -p /n/fs/aa-rldiff/winter/s-flm/data_cache

cd /n/fs/aa-rldiff/winter/s-flm

CKPT_PATH=/n/fs/aa-rldiff/winter/s-flm/checkpoints/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt
CACHE_DIR=/n/fs/aa-rldiff/winter/s-flm/data_cache
OUTPUT_DIR=/n/fs/aa-rldiff/winter/s-flm/outputs/eval_psi_${TAG}_${SLURM_JOB_ID}
STEPS=32

echo "=== GSM8K eval: tag=${TAG}, psi=${PSI_CKPT}, scale=${PSI_SCALE} ==="
hostname; nvidia-smi | head -10 || true

python -u -m main \
    mode=gsm8k_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    data=gsm8k-test \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir="${CACHE_DIR}" \
    data.data_path=/n/fs/aa-rldiff/winter/s-flm/data/gsm8k_test.json \
    model=small-sphere-arch \
    model.length=512 \
    model.normalize_input_embed=False \
    algo=sfm \
    algo.renormalize_weights=True \
    algo.invert_time_convention=false \
    noise=log-linear-adaptive \
    noise.alpha_max=0.121 \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=25600 \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    sampler=sfm \
    sampler.predictor=sfm_psi \
    sampler.noise_removal=greedy \
    sampler.velocity=exact \
    sampler.top_k_velocity=-1 \
    sampler.steps="${STEPS}" \
    loader.eval_batch_size=16 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    gsm8k.output_dir="${OUTPUT_DIR}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"

echo ""
echo "=== DONE: tag=${TAG}, scale=${PSI_SCALE} ==="
