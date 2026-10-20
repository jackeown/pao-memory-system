#!/usr/bin/env bash
# Generate every missing PAO card image (images/00.png ... images/99.png) with
# the local Stable Diffusion install in .venv. Safe to re-run: existing files
# are skipped, so it resumes where it left off.
#
#   ./generate_all.sh                 # everything missing
#   ./generate_all.sh --force         # regenerate all
#   ./generate_all.sh --only 40,88    # just these numbers
set -euo pipefail
cd "$(dirname "$0")"

exec .venv/bin/python generate_images.py \
  --provider diffusers \
  --model Lykon/dreamshaper-8 \
  --width 512 --height 512 \
  --steps 28 \
  --cfg 7 \
  --no-safety-checker \
  "$@"
