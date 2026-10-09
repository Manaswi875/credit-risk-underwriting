#!/usr/bin/env bash
# Reproduce the full pipeline: download -> prepare -> train -> analyze -> calibrate -> fair lending.
set -euo pipefail
cd "$(dirname "$0")"
RAW=data/raw/accepted_2007_to_2018Q4.csv
if [ ! -f "$RAW" ]; then
  mkdir -p data/raw
  echo "Downloading LendingClub accepted loans (~1.6 GB)..."
  curl -L -o "$RAW" \
    "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv"
fi
ACS=https://www2.census.gov/programs-surveys/acs/summary_file/2021/table-based-SF/data/5YRData
mkdir -p data/external
for tbl in b03002 b19013; do
  f=data/external/acsdt5y2021-$tbl.dat
  [ -f "$f" ] || curl -L -o "$f" "$ACS/acsdt5y2021-$tbl.dat"
done
mkdir -p models reports
cd src
python prepare_data.py
python train.py
python analyze.py
python calibrate.py
python fair_lending.py
