#!/usr/bin/env bash
# Create the venv and check the system deps. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "no $PY on PATH"; exit 1; }
echo "python:  $("$PY" --version)"

for bin in ffmpeg ffprobe; do
  if command -v "$bin" >/dev/null; then
    echo "$bin:  $("$bin" -version 2>&1 | head -1 | cut -c1-40)"
  else
    echo "MISSING: $bin -- brew install ffmpeg  /  apt install ffmpeg"; exit 1
  fi
done

[ -d .venv ] || { echo "creating .venv"; "$PY" -m venv .venv; }
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install --quiet -r requirements.txt
./.venv/bin/python - <<'PYCHK'
import torch
from net import pick_device
dev = pick_device()
extra = ""
if dev == "cuda":
    extra = f"  ({torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda})"
elif dev == "cpu":
    extra = "  -- no GPU found; training will be slow"
print(f"torch {torch.__version__}, device {dev}{extra}")
PYCHK

mkdir -p video/out
shopt -s nullglob
clips=(video/*.mp4)
if [ ${#clips[@]} -eq 0 ]; then
  cat <<'MSG'

Next: drop the source clip in  video/   (any .mp4; the first one found is used)
Then:
  ./.venv/bin/python generate.py --loss all
MSG
else
  echo
  echo "source clip: ${clips[0]}"
  echo "next:  ./.venv/bin/python generate.py --loss all"
fi
