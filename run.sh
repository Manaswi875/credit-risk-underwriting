#!/usr/bin/env bash
# Reproduce the full pipeline: download -> prepare -> train -> analyze.
set -euo pipefail
cd "$(dirname "$0")"
RAW=data/raw/accepted_2007_to_2018Q4.csv
if [ ! -f "$RAW" ]; then
  mkdir -p data/raw
  echo "Downloading LendingClub accepted loans (~1.6 GB)..."
  curl -L -o "$RAW" \
    "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv"
fi
mkdir -p models reports
cd src
python prepare_data.py
python train.py
python analyze.py
