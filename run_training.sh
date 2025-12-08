#!/bin/bash
set -euo pipefail

export HF_HOME=""
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


export HF_HUB_OFFLINE=1
unset CUDA_VISIBLE_DEVICES

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
TASK_NAME="zebra-cot"
EPOCHS=15
GRAD_ACCUM_STEPS=8
LATENT_SIZE=8
CE_WEIGHT=1
WARM_UP_STEPS=
SAVE_STEPS=
DATA_PATH=""
SAVE_MODEL_PATH=""
LOG_FILE=""


mkdir -p "$(dirname "$SAVE_MODEL_PATH")" "$(dirname "$LOG_FILE")"


NUM_PROCESSES=${NUM_PROCESSES:-$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() or 1)
except Exception:
    print(1)
PY
)}
echo "Using ${NUM_PROCESSES} processes"

accelerate launch --num_processes="${NUM_PROCESSES}" src/main.py \
  --model "${MODEL_NAME}" \
  --epochs "${EPOCHS}" \
  --task "${TASK_NAME}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --stage stage1 \
  --warm_up_steps "${WARM_UP_STEPS}" \
  --data_path "${DATA_PATH}" \
  --log_file "${LOG_FILE}" \
  --latent_size "${LATENT_SIZE}" \
  --ce_weight "${CE_WEIGHT}" \
  --save_model_path "${SAVE_MODEL_PATH}" \
  --cache_dir "${HF_HOME}" \
  --save_steps "${SAVE_STEPS}"

echo "training finished, save model to ${SAVE_MODEL_PATH}"
