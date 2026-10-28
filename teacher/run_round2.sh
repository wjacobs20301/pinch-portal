#!/usr/bin/env bash
# Round 2: bigger student (width 32) that repaints the whole frame, trained with an adversarial critic.
# Rerun with --resume appended to continue after an interruption.
cd "$(dirname "$0")"
LOG=logs_round2.txt
echo "$(date) starting round 2" >> $LOG
systemd-inhibit --what=sleep:idle --who=filter-app --why="Training anime model" \
  .venv/bin/python train.py --name student_v2 --width 32 --no-residual --gan --iters 40000 --batch 16 "$@" 2>&1 \
  | grep -E --line-buffered "pairs|iter [0-9]*000/|Traceback|Error" >> $LOG
echo "$(date) round 2 finished" >> $LOG
