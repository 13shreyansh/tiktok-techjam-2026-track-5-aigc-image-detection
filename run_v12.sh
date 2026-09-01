#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 IMAGE_DIRECTORY OUTPUT_JSON [PE_CHECKPOINT] [DINO_CHECKPOINT]" >&2
  exit 2
fi

image_directory=$1
output_json=$2
pe_checkpoint=${3:-models/v12_pe_core.pt}
dino_checkpoint=${4:-models/v12_dinov2.pt}
python_executable=${PYTHON_EXECUTABLE:-python3}
script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$script_directory/src${PYTHONPATH:+:$PYTHONPATH}"

arguments=(
  "$image_directory"
  --pe-checkpoint "$pe_checkpoint"
  --output "$output_json"
  --device "${AIGC_DEVICE:-auto}"
  --mode "${AIGC_V12_MODE:-pe_core}"
  --batch-size "${AIGC_BATCH_SIZE:-1}"
)

if [[ ${AIGC_V12_MODE:-pe_core} == blend ]]; then
  arguments+=(--dino-checkpoint "$dino_checkpoint")
fi

exec "$python_executable" -m aigc_detector.predict_v12 "${arguments[@]}"
