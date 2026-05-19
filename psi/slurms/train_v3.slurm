#!/bin/bash
#SBATCH --job-name=psi_c_rl
#SBATCH --output=/n/fs/aa-rldiff/winter/s-flm/outputs/train_psi_c_%j.out
#SBATCH --error=/n/fs/aa-rldiff/winter/s-flm/outputs/train_psi_c_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:l40:1

# Variant C / B': REINFORCE on a hard-circulation-projected psi.
# psi lives in {tangent at z} ∩ {orthogonal to tangent score} by
# architectural projection (CirculationPsiNet). We optimize the
# expected GSM8K verifier reward via REINFORCE with per-prompt-mean
# baseline.

set -uo pipefail

export TMPDIR=/n/fs/aa-rldiff/tmp
export XDG_CACHE_HOME=/n/fs/aa-rldiff/.cache
export HF_HOME=/n/fs/aa-rldiff/hf_cache_global
export TRANSFORMERS_CACHE=/n/fs/aa-rldiff/hf_cache_global
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false

source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda/envs/drifting

mkdir -p /n/fs/aa-rldiff/winter/s-flm/outputs
mkdir -p /n/fs/aa-rldiff/winter/s-flm/outputs/psi_c_rl

cd /n/fs/aa-rldiff/winter/s-flm

echo "=== Variant C: REINFORCE on circulation psi (GSM8K verifier) ==="
hostname; nvidia-smi | head -10 || true

CKPT_PATH=/n/fs/aa-rldiff/winter/s-flm/checkpoints/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt
OUTPUT_DIR=/n/fs/aa-rldiff/winter/s-flm/outputs/psi_c_rl

python -u -m main \
    mode=psi_train_c \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.disable_ema=false \
    data=tiny-gsm \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=/n/fs/aa-rldiff/winter/s-flm/data_cache_psi \
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
    loader.batch_size=2 \
    loader.eval_batch_size=8 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    +psi.n_steps=1500 \
    +psi.gsm8k_train_cache=/n/fs/aa-rldiff/winter/s-flm/data/gsm8k_test_as_train.json \
    +psi.use_internal_sampler=true \
    +psi.lr=1e-4 \
    +psi.hidden=512 \
    +psi.prompt_batch=2 \
    +psi.K_rollouts=4 \
    +psi.sampler_steps=32 \
    +psi.sigma_explore=0.0 \
    +psi.max_prompt_len=192 \
    +psi.verifier_timeout_s=2.0 \
    +psi.verifier_workers=8 \
    +psi.grad_clip=1.0 \
    +psi.log_every=10 \
    +psi.save_every=100 \
    +psi.out_dir="${OUTPUT_DIR}" \
    gsm8k.output_dir="${OUTPUT_DIR}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"

echo ""
echo "=== DONE -- psi (C) ckpts at ${OUTPUT_DIR}/psi_*.pt ==="
