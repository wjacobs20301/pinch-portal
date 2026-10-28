#!/usr/bin/env bash
# Round 5: browser-speed model (~20 ms/frame → 30 fps). A FastStudent trained from scratch to reproduce the smooth
# round 4 model (distillation) and the teacher's anime, with the critic and anti-flicker loss.
# Waits for round 4 to finish first. Rerunnable: resumes from its last checkpoint.
cd "$(dirname "$0")"
LOG=logs_round5.txt
while tmux has-session -t round4 2>/dev/null; do sleep 60; done
echo "$(date) round 5: fast model, distilling from v4" >> $LOG
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True systemd-inhibit --what=sleep:idle --who=filter-app --why="Training anime model" \
  .venv/bin/python train.py --name student_v5_fast --arch fast --width 48 --gan --gan-start 5000 \
  --data data/pairs_v3,data/faces_v3 --distill checkpoints/student_v4.pt --temporal 3 \
  --iters 100000 --batch 6 --crop 320 --lr 2e-4 --resume 2>&1 \
  | grep -E --line-buffered "pairs|distilling|iter [0-9]*000/|Traceback|Error" >> $LOG
echo "$(date) round 5 finished" >> $LOG
