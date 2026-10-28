#!/usr/bin/env bash
# Round 3: stronger anime teacher + face close-ups, then a long (~20 h) training run initialised from v2.
# Rerunnable: the teacher skips finished photos and training resumes from its last checkpoint.
cd "$(dirname "$0")"
LOG=logs_round3.txt
AWAKE=(systemd-inhibit --what=sleep:idle --who=filter-app --why="Training anime model")
echo "$(date) round 3: teacher on full scenes (setting C)" >> $LOG
"${AWAKE[@]}" .venv/bin/python generate_pairs.py --src data/raw --dst data/pairs_v3 \
  --strength 0.70 --cn 0.8 --cn-end 0.6 2>&1 | grep -E --line-buffered "photos to process|00/|Traceback|Error:" >> $LOG
echo "$(date) teacher on face close-ups (setting D)" >> $LOG
"${AWAKE[@]}" .venv/bin/python generate_pairs.py --src data/raw_faces --dst data/faces_v3 \
  --strength 0.75 --cn 0.7 --cn-end 0.5 2>&1 | grep -E --line-buffered "photos to process|00/|Traceback|Error:" >> $LOG
echo "$(date) training v3" >> $LOG
"${AWAKE[@]}" .venv/bin/python train.py --name student_v3 --width 32 --no-residual --gan \
  --data data/pairs_v3,data/faces_v3 --init checkpoints/student_v2.pt --gan-start 0 \
  --iters 150000 --batch 10 --crop 320 --lr 1e-4 --resume 2>&1 \
  | grep -E --line-buffered "pairs|initialised|iter [0-9]*000/|Traceback|Error" >> $LOG
echo "$(date) round 3 finished" >> $LOG
