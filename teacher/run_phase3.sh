#!/usr/bin/env bash
# Phase 3: the sharp live model (FastStudent2x, 448x336 → 896x672) on phase 2's data, ~1 day. Waits for phase 2.
# Starts with a short trial run and picks the largest batch size that fits in GPU memory (round 5 lost a night
# to an out-of-memory crash). Rerunnable: resumes from its checkpoint.
cd "$(dirname "$0")"
LOG=logs_phase3.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while tmux has-session -t phase2 2>/dev/null; do sleep 60; done
echo "$(date) finding mouths (lip landmarks) for mouth-weighted training" >> $LOG
.venv/bin/python make_mouths.py data/pairs_v7c data/faces_v7c 2>&1 | grep -E "mouths found|Traceback|Error" >> $LOG
COMMON=(--data data/pairs_v7c,data/faces_v7c --init checkpoints/student_v6_fast.pt --iters 200000)
BATCH=""
for b in 6 4 3; do
  if timeout 900 .venv/bin/python train_hr.py --name preflight --batch $b --iters 20 --gan-start 0 \
       --data data/pairs_v7c,data/faces_v7c > /tmp/preflight.log 2>&1; then BATCH=$b; break; fi
  echo "$(date) batch $b failed: $(grep -m1 -E 'Error|error' /tmp/preflight.log)" >> $LOG
done
rm -f checkpoints/preflight*
[ -z "$BATCH" ] && { echo "$(date) phase 3: no batch size fits, stopping" >> $LOG; exit 1; }
echo "$(date) phase 3: training with batch $BATCH" >> $LOG
systemd-inhibit --what=sleep:idle --who=filter-app --why="Training anime model" \
  .venv/bin/python train_hr.py "${COMMON[@]}" --batch $BATCH --resume 2>&1 \
  | grep -E --line-buffered "pairs|initialised|iter [0-9]*000/|Traceback|Error" >> $LOG
echo "$(date) phase 3 finished" >> $LOG
