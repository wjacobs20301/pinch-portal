#!/usr/bin/env bash
# Round 6: fine-tune the fast model on colour-corrected pairs (teacher drawings with the photo's real hair/skin/
# clothes colours, see fix_colors.py). Waits for round 5. Uses less GPU memory than round 5 (no distill mentor).
cd "$(dirname "$0")"
LOG=logs_round6.txt
while tmux has-session -t round5 2>/dev/null; do sleep 60; done
echo "$(date) round 6: fast model on colour-corrected data, from v5" >> $LOG
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True systemd-inhibit --what=sleep:idle --who=filter-app --why="Training anime model" \
  .venv/bin/python train.py --name student_v6_fast --arch fast --width 48 --gan --gan-start 0 \
  --data data/pairs_v6,data/faces_v6 --init checkpoints/student_v5_fast.pt --temporal 3 \
  --iters 40000 --batch 6 --crop 320 --lr 1e-4 --resume 2>&1 \
  | grep -E --line-buffered "pairs|initialised|iter [0-9]*000/|Traceback|Error" >> $LOG
echo "$(date) round 6 finished" >> $LOG
