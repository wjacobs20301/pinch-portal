#!/usr/bin/env bash
# Phase 2: full-quality, face-focused teacher data (~21 h on the 3070). Rerunnable: every step skips finished work.
cd "$(dirname "$0")"
LOG=logs_phase2.txt
AWAKE=(systemd-inhibit --what=sleep:idle --who=filter-app --why="Generating anime training data")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while tmux has-session -t originals 2>/dev/null; do sleep 60; done
echo "$(date) phase 2: $(ls data/raw_full | wc -l) full-res originals" >> $LOG
.venv/bin/python make_faces.py data/raw_full data/raw_faces_full 896 >> $LOG 2>&1
echo "$(date) teacher: face close-ups" >> $LOG
"${AWAKE[@]}" .venv/bin/python generate_pairs_v7.py --src data/raw_faces_full --dst data/faces_v7 \
  --strength 0.75 --cn 0.7 --cn-end 0.5 2>&1 | grep -E --line-buffered "photos to process|00/|failed|Traceback|Error:" >> $LOG
echo "$(date) teacher: scenes with face detail" >> $LOG
"${AWAKE[@]}" .venv/bin/python generate_pairs_v7.py --src data/raw --dst data/pairs_v7 --face-detail 2>&1 \
  | grep -E --line-buffered "photos to process|00/|failed|Traceback|Error:" >> $LOG
echo "$(date) colour fix (anime eyes kept)" >> $LOG
.venv/bin/python fix_colors.py data/faces_v7 data/faces_v7c --keep-eyes 2>&1 | grep -E "done|Traceback|Error" >> $LOG
.venv/bin/python fix_colors.py data/pairs_v7 data/pairs_v7c --keep-eyes 2>&1 | grep -E "done|Traceback|Error" >> $LOG
echo "$(date) phase 2 finished" >> $LOG
