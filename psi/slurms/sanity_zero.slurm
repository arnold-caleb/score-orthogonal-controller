#!/bin/bash
#SBATCH --job-name=sflm_psi_sanity
#SBATCH --output=/n/fs/aa-rldiff/winter/s-flm/outputs/sanity_psi_zero_%j.out
#SBATCH --error=/n/fs/aa-rldiff/winter/s-flm/outputs/sanity_psi_zero_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:l40:1

# Phase-2 step-1 sanity: run GSM8K eval with sfm_psi sampler but ZERO-init psi.
# Expect: identical accuracy to vanilla SFM (12.51%, 165/1319). If yes,
# integration is correct and we can start training the psi residual.

set -uo pipefail

export TMPDIR=/n/fs/aa-rldiff/tmp
export XDG_CACHE_HOME=/n/fs/aa-rldiff/.cache
export HF_HOME=/n/fs/aa-rldiff/hf_cache_global
export TRANSFORMERS_CACHE=/n/fs/aa-rldiff/hf_cache_global
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Tell the sfm_psi factory: build a 768-d SphereSolenoidalNet, no checkpoint.
# Zero-init -> psi == 0 -> sampler reduces to vanilla SFM.
export SFLM_PSI_D=768
export SFLM_PSI_HIDDEN=512
unset SFLM_PSI_CKPT

source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda/envs/drifting

mkdir -p /n/fs/aa-rldiff/winter/s-flm/outputs
mkdir -p /n/fs/aa-rldiff/winter/s-flm/data_cache

cd /n/fs/aa-rldiff/winter/s-flm

echo "=== S-FLM TinyGSM with sfm_psi (zero-init psi) on GSM8K -- SANITY ==="
hostname; nvidia-smi | head -10 || true

CKPT_PATH=/n/fs/aa-rldiff/winter/s-flm/checkpoints/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt
CACHE_DIR=/n/fs/aa-rldiff/winter/s-flm/data_cache
OUTPUT_DIR=/n/fs/aa-rldiff/winter/s-flm/outputs/sanity_psi_zero_eval
STEPS=32

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
echo "=== DONE -- check that accuracy == 12.51% (165/1319) ==="
