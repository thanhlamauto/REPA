#!/usr/bin/env bash
# =============================================================================
# REPA LoRA Fine-tuning Experiment — EuroSAT (10 classes)
#
# Comparison:
#   Run A — REPA pretrained ckpt  +  fine-tune WITHOUT REPA alignment loss
#   Run B — REPA pretrained ckpt  +  fine-tune WITH    REPA alignment loss (λ=0.1)
#
# Each run generates 500 images/class → 1000 images/class total.
#
# Usage:
#   bash run_eurosat_experiment.sh
#   bash run_eurosat_experiment.sh 2>&1 | tee experiment.log
#
# Override any variable via environment, e.g.:
#   EPOCHS=200 bash run_eurosat_experiment.sh
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Paths (all relative to repo root)
# --------------------------------------------------------------------------- #
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_DIR}/data/eurosat"
LORA_A_DIR="${REPO_DIR}/lora_A"
LORA_B_DIR="${REPO_DIR}/lora_B"
SYNTH_A_DIR="${REPO_DIR}/synthetic_A"
SYNTH_B_DIR="${REPO_DIR}/synthetic_B"
PREVIEW_DIR="${REPO_DIR}/previews"

# --------------------------------------------------------------------------- #
# Hyper-parameters (override via env vars)
# --------------------------------------------------------------------------- #
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
REPA_COEFF="${REPA_COEFF:-0.1}"
ENC_TYPE="${ENC_TYPE:-dinov2-vit-b}"

N_SAMPLES="${N_SAMPLES:-500}"
N_SANITY="${N_SANITY:-32}"
SAMPLING_BATCH="${SAMPLING_BATCH:-32}"
NUM_STEPS="${NUM_STEPS:-50}"
SAMPLING_MODE="${SAMPLING_MODE:-sde}"

CFG_SCALE_A="${CFG_SCALE_A:-2.0}"   # Run A: no REPA — slightly higher guidance
CFG_SCALE_B="${CFG_SCALE_B:-1.4}"   # Run B: REPA   — lower guidance works better

SEED="${SEED:-0}"

# --------------------------------------------------------------------------- #
# EuroSAT classes (index → name)
# --------------------------------------------------------------------------- #
EUROSAT_CLASSES=(
    "AnnualCrop"
    "Forest"
    "HerbaceousVegetation"
    "Highway"
    "Industrial"
    "Pasture"
    "PermanentCrop"
    "Residential"
    "River"
    "SeaLake"
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "WARNING: nvidia-smi not found — running on CPU will be very slow."
    else
        log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
    fi
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
cd "${REPO_DIR}"

log "======================================================================"
log "REPA LoRA Experiment — EuroSAT  |  epochs=${EPOCHS}  seed=${SEED}"
log "  Run A: v-pred only  (cfg=${CFG_SCALE_A})"
log "  Run B: v-pred + REPA loss λ=${REPA_COEFF}  (cfg=${CFG_SCALE_B})"
log "======================================================================"

check_gpu

# ------------------------------------------------------------------ #
# Step 0 — Prepare data
# ------------------------------------------------------------------ #
log ">>> Step 0: Preparing EuroSAT few-shot data"
python prepare_fewshot_data.py \
    --raw-dir  "data/eurosat_raw" \
    --data-dir "${DATA_DIR}" \
    --n-shot   16 \
    --seed     "${SEED}"
log "Data ready."

# ------------------------------------------------------------------ #
# Per-class loop
# ------------------------------------------------------------------ #
for class_idx in "${!EUROSAT_CLASSES[@]}"; do
    class_name="${EUROSAT_CLASSES[$class_idx]}"

    log ""
    log "======================================================================"
    log "Class ${class_idx}/9 : ${class_name}"
    log "======================================================================"

    # ------------------------------------------------------------ #
    # Run A — fine-tune WITHOUT REPA loss
    # ------------------------------------------------------------ #
    log "[A] Fine-tuning (v-pred only, no REPA loss) …"
    python finetune_lora.py \
        --data-dir    "${DATA_DIR}" \
        --class-idx   "${class_idx}" \
        --class-name  "${class_name}" \
        --output-dir  "${LORA_A_DIR}" \
        --lora-rank   "${LORA_RANK}" \
        --epochs      "${EPOCHS}" \
        --batch-size  "${BATCH_SIZE}" \
        --lr          "${LR}" \
        --fp16 \
        --cfg-prob    0.1 \
        --seed        "${SEED}"

    log "[A] Generating ${N_SAMPLES} images (cfg=${CFG_SCALE_A}) …"
    python generate_lora.py \
        --class-idx   "${class_idx}" \
        --class-name  "${class_name}" \
        --lora-path   "${LORA_A_DIR}/${class_name}.safetensors" \
        --output-dir  "${SYNTH_A_DIR}" \
        --preview-dir "${PREVIEW_DIR}" \
        --n-samples   "${N_SAMPLES}" \
        --n-sanity    "${N_SANITY}" \
        --cfg-scale   "${CFG_SCALE_A}" \
        --num-steps   "${NUM_STEPS}" \
        --mode        "${SAMPLING_MODE}" \
        --batch-size  "${SAMPLING_BATCH}" \
        --lora-rank   "${LORA_RANK}" \
        --seed        "${SEED}"

    # ------------------------------------------------------------ #
    # Run B — fine-tune WITH REPA alignment loss
    # ------------------------------------------------------------ #
    log "[B] Fine-tuning (v-pred + REPA loss λ=${REPA_COEFF}) …"
    python finetune_lora.py \
        --data-dir    "${DATA_DIR}" \
        --class-idx   "${class_idx}" \
        --class-name  "${class_name}" \
        --output-dir  "${LORA_B_DIR}" \
        --lora-rank   "${LORA_RANK}" \
        --epochs      "${EPOCHS}" \
        --batch-size  "${BATCH_SIZE}" \
        --lr          "${LR}" \
        --fp16 \
        --cfg-prob    0.1 \
        --use-repa \
        --repa-coeff  "${REPA_COEFF}" \
        --enc-type    "${ENC_TYPE}" \
        --seed        "${SEED}"

    log "[B] Generating ${N_SAMPLES} images (cfg=${CFG_SCALE_B}) …"
    python generate_lora.py \
        --class-idx   "${class_idx}" \
        --class-name  "${class_name}" \
        --lora-path   "${LORA_B_DIR}/${class_name}.safetensors" \
        --output-dir  "${SYNTH_B_DIR}" \
        --preview-dir "${PREVIEW_DIR}" \
        --n-samples   "${N_SAMPLES}" \
        --n-sanity    "${N_SANITY}" \
        --cfg-scale   "${CFG_SCALE_B}" \
        --num-steps   "${NUM_STEPS}" \
        --mode        "${SAMPLING_MODE}" \
        --batch-size  "${SAMPLING_BATCH}" \
        --lora-rank   "${LORA_RANK}" \
        --seed        "${SEED}"

    log "Class ${class_name} done."
done

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
log ""
log "======================================================================"
log "Experiment complete!"
log "  LoRA A weights : ${LORA_A_DIR}/"
log "  LoRA B weights : ${LORA_B_DIR}/"
log "  Synthetic A    : ${SYNTH_A_DIR}/  (no REPA loss)"
log "  Synthetic B    : ${SYNTH_B_DIR}/  (with REPA loss)"
log "  Preview grids  : ${PREVIEW_DIR}/"
log ""
log "Next steps:"
log "  1. Visually inspect previews/ grids"
log "  2. Compute FID with clean-fid:"
log "       pip install clean-fid"
log "       python -m cleanfid.fid --dir1 synthetic_A/<class> --dir2 data/eurosat/real_train_fewshot/seed0/<class>"
log "  3. Compare denoising loss curves in lora_A/*_log.json vs lora_B/*_log.json"
log "======================================================================"
