#!/usr/bin/env bash
# Lay out the directories the author's notebooks expect, link the six shared
# *_labeled_10m_done.csv files, and run the NB2 + NB3 prep replay (CPU image).
# Usage: run_prep.sh <workdir> <dir with *_done.csv> <seaice_cpu.sif>
set -eu
W=$1; DATA=$2; SIF=$3
B="$W/ATL03_labeled_data/cor_labeled_by_is2trackline_10m"
mkdir -p "$B/done" "$B/relative_height" "$B/relative_height_interpolated" \
         "$B/relative_height_interpolated_h_diff" "$W/ATL03_labeled_data/cor_labeled_LSTM"
ln -sf "$(cd "$DATA" && pwd)"/*_done.csv "$B/done/"
cp "$(dirname "$0")"/replay_prep.py "$W/"
cd "$W" && apptainer exec "$SIF" python3 replay_prep.py > prep.log 2>&1
