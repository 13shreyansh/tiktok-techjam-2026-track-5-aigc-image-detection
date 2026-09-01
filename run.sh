#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 IMAGE_DIRECTORY OUTPUT_JSON [CHECKPOINT]" >&2
  exit 2
fi

image_directory=$1
output_json=$2
checkpoint=${3:-models/model.pt}
python_executable=${PYTHON_EXECUTABLE:-python3}
script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$script_directory/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_executable" -m aigc_detector.predict \
  "$image_directory" \
  --checkpoint "$checkpoint" \
  --output "$output_json" \
  --device "${AIGC_DEVICE:-auto}"
