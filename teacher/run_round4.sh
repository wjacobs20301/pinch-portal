#!/usr/bin/env bash
# Round 4: anti-flicker. Continue from v3 with a consistency loss (a slightly shifted/noisier/brighter copy of each
# frame must be drawn the same way), same anime data and critic. Rerunnable: resumes from its last checkpoint.
cd "$(dirname "$0")"
LOG=logs_round4.txt
echo "$(date) round 4: anti-flicker training from v3" >> $LOG
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True systemd-inhibit --what=sleep:idle --who=filter-app --why="Training anime model" \
  .venv/bin/python train.py --name student_v4 --width 32 --no-residual --gan --gan-start 0 \
  --data data/pairs_v3,data/faces_v3 --init checkpoints/student_v3.pt --temporal 3 \
  --iters 80000 --batch 6 --crop 320 --lr 1e-4 --resume 2>&1 \
  | grep -E --line-buffered "pairs|initialised|iter [0-9]*000/|Traceback|Error" >> $LOG
echo "$(date) round 4 finished" >> $LOG
