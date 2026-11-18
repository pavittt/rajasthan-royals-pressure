#!/usr/bin/env bash
# Rebuilds every output from the supplied data pack. Usage: RAW_DIR=/path/to/data ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data out
for s in s01_prepare s02_winprob s03_pressure s04_ball_value s00_rr_moments s00b_rr_pressure s05_ratings s06_reliability s07_sensitivity s09_lists s09b_form s12_plan s12b_moves s10_page; do
  echo "=== $s"; python3 "$s.py" > "out/log_$s.txt" 2>&1 || { echo "FAILED: $s (see out/log_$s.txt)"; exit 1; }
done
google-chrome --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer --virtual-time-budget=6000 --print-to-pdf=out/SupeRR_Round1_Report.pdf "file://$PWD/out/dashboard.html?print=1"
python3 - <<'P'
import hashlib, glob
for f in sorted(glob.glob("out/official_top10_*.csv")):
    print(f, hashlib.md5(open(f, "rb").read()).hexdigest())
P
