#!/usr/bin/env bash
# Full unattended pipeline: teacher redraws every photo, then the live student model trains on the results.
# Safe to rerun: the teacher skips photos it already did, and training resumes from its last checkpoint.
# Works on the Linux desktop (systemd-inhibit keeps it awake) and on the Mac (caffeinate).
cd "$(dirname "$0")"
LOG=logs_generate.txt

if command -v systemd-inhibit >/dev/null 2>&1; then
  AWAKE=(systemd-inhibit --what=sleep:idle --who=filter-app --why="Training anime model")
elif command -v caffeinate >/dev/null 2>&1; then
  AWAKE=(caffeinate -i)
else
  AWAKE=()
fi

while pgrep -f "fetch_data.py" >/dev/null; do sleep 10; done
echo "$(date) starting teacher on $(ls data/raw | wc -l) photos" >> $LOG
"${AWAKE[@]}" .venv/bin/python generate_pairs.py --size 640 --steps 8 2>&1 \
  | grep -E --line-buffered "photos to process|each|failed|Traceback|Error:" >> $LOG
echo "$(date) teacher finished: $(ls data/pairs/*_out.jpg | wc -l) pairs" >> $LOG

RESUME=""; [ -f checkpoints/student.pt ] && RESUME="--resume"
echo "$(date) starting training $RESUME" >> $LOG
"${AWAKE[@]}" .venv/bin/python train.py --batch 16 --iters 30000 $RESUME 2>&1 \
  | grep -E --line-buffered "pairs|iter [0-9]*000/|Traceback|Error:" >> $LOG
echo "$(date) training finished" >> $LOG
