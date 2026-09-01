#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 IMAGE_DIRECTORY OUTPUT_JSON [V6_CHECKPOINT] [V9_CHECKPOINT]" >&2
  exit 2
fi

image_directory=$1
output_json=$2
v6_checkpoint=${3:-models/model.pt}
v9_checkpoint=${4:-models/model_v9.pt}
python_executable=${PYTHON_EXECUTABLE:-python3}
script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$script_directory/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_executable" -m aigc_detector.predict_ensemble \
  "$image_directory" \
  --v6-checkpoint "$v6_checkpoint" \
  --v9-checkpoint "$v9_checkpoint" \
  --output "$output_json" \
  --device "${AIGC_DEVICE:-auto}"
