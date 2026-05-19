#!/bin/bash
#SBATCH --job-name=psi_v2B
#SBATCH --output=/n/fs/aa-rldiff/winter/s-flm/outputs/train_psi_v2B_%j.out
#SBATCH --error=/n/fs/aa-rldiff/winter/s-flm/outputs/train_psi_v2B_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:l40:1

# Variant B: A + tangent-divergence penalty.
# Tests whether the divergence-free (circulation) subspace of the frozen
# flow is a privileged inference-time control subspace for reasoning.

set -uo pipefail

export TMPDIR=/n/fs/aa-rldiff/tmp
export XDG_CACHE_HOME=/n/fs/aa-rldiff/.cache
export HF_HOME=/n/fs/aa-rldiff/hf_cache_global
export TRANSFORMERS_CACHE=/n/fs/aa-rldiff/hf_cache_global
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda/envs/drifting

mkdir -p /n/fs/aa-rldiff/winter/s-flm/outputs
mkdir -p /n/fs/aa-rldiff/winter/s-flm/data_cache_psi
mkdir -p /n/fs/aa-rldiff/winter/s-flm/outputs/psi_v2B

cd /n/fs/aa-rldiff/winter/s-flm

echo "=== Variant B: supervised + tangent-divergence penalty (TinyGSM) ==="
hostname; nvidia-smi | head -10 || true

CKPT_PATH=/n/fs/aa-rldiff/winter/s-flm/checkpoints/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt
OUTPUT_DIR=/n/fs/aa-rldiff/winter/s-flm/outputs/psi_v2B
PSI_INIT=/n/fs/aa-rldiff/winter/s-flm/outputs/psi_v1A/psi_step8000.pt

python -u -m main \
    mode=psi_train_v2 \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.disable_ema=true \
    data=tiny-gsm \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=/n/fs/aa-rldiff/winter/s-flm/data_cache_psi \
    data.train_on_prompt=False \
    data.wrap=False \
    data.filter_too_long=True \
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
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    +psi.n_steps=8000 \
    +psi.lr=3e-4 \
    +psi.weight_decay=0.0 \
    +psi.hidden=512 \
    +psi.alpha_min=0.05 \
    +psi.alpha_max=0.95 \
    +psi.lambda_div=0.1 \
    +psi.n_div_probes=4 \
    +psi.init_from="${PSI_INIT}" \
    +psi.log_every=50 \
    +psi.save_every=500 \
    +psi.dataset_mode=gsm8k_train \
    +psi.out_dir="${OUTPUT_DIR}" \
    gsm8k.output_dir="${OUTPUT_DIR}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"

echo ""
echo "=== DONE -- psi (B) ckpts at ${OUTPUT_DIR}/psi_*.pt ==="
