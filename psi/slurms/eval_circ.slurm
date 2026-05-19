#!/bin/bash
#SBATCH --job-name=sflm_eval_c
#SBATCH --output=/n/fs/aa-rldiff/winter/s-flm/outputs/eval_with_psi_circ_%j.out
#SBATCH --error=/n/fs/aa-rldiff/winter/s-flm/outputs/eval_with_psi_circ_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:l40:1

# GSM8K eval through SFMSamplerWithPsiCirc using a trained CirculationPsiNet.
# Env vars:
#   PSI_CKPT   - path to trained psi .pt
#   TAG        - short label (e.g. 'C_step1500')
#   PSI_SCALE  - optional scale (default 1.0); useful for ablation

set -uo pipefail

: "${PSI_CKPT:?Set PSI_CKPT}"
: "${TAG:?Set TAG}"
PSI_SCALE="${PSI_SCALE:-1.0}"

export TMPDIR=/n/fs/aa-rldiff/tmp
export XDG_CACHE_HOME=/n/fs/aa-rldiff/.cache
export HF_HOME=/n/fs/aa-rldiff/hf_cache_global
export TRANSFORMERS_CACHE=/n/fs/aa-rldiff/hf_cache_global
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false

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
OUTPUT_DIR=/n/fs/aa-rldiff/winter/s-flm/outputs/eval_psi_circ_${TAG}_${SLURM_JOB_ID}
STEPS=32

echo "=== GSM8K eval (circ psi): tag=${TAG}, psi=${PSI_CKPT}, scale=${PSI_SCALE} ==="
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
    sampler.predictor=sfm_psi_circ \
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
echo "=== DONE: tag=${TAG} ==="
