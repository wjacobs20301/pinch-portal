#!/usr/bin/env bash
# Phase 4: the dedicated face model. ~18 h. Every step resumes where it left off (runs as filter-face.service).
#  1. expression-locked teacher redraws all face close-ups (lineart + MediaPipe-face ControlNets, anti-frown)
#  2. real colours (anime eyes kept) + brighter, softer lighting + alignment to 256x256 (eyes/mouth at fixed spots)
#  3. train FaceStudent, mouth weighted 6x; preflight picks a batch size that fits in GPU memory
cd "$(dirname "$0")"
LOG=logs_phase4.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AWAKE=(systemd-inhibit --what=sleep:idle --who=filter-app --why="Training anime face model")
echo "$(date) phase 4: expression-locked teacher on face close-ups" >> $LOG
"${AWAKE[@]}" .venv/bin/python teacher_face.py --src data/raw_faces_full --dst data/faces_v9 --face 1.1 2>&1 \
  | grep -E --line-buffered "photos to process|00/|failed|Traceback|Error:" >> $LOG
echo "$(date) colours + lighting + alignment" >> $LOG
.venv/bin/python align_faces.py data/faces_v9 --out data/face_aligned_v9 2>&1 | grep -E "aligned|Traceback|Error" >> $LOG
BATCH=""
for b in 16 12 8; do
  if timeout 900 .venv/bin/python train_face.py --name preflight --batch $b --iters 20 --gan-start 0 \
       --data data/face_aligned_v9 > /tmp/preflight_face.log 2>&1; then BATCH=$b; break; fi
  echo "$(date) batch $b failed: $(grep -m1 -E 'Error|error' /tmp/preflight_face.log)" >> $LOG
done
rm -f checkpoints/preflight*
[ -z "$BATCH" ] && { echo "$(date) phase 4: no batch size fits, stopping" >> $LOG; exit 1; }
echo "$(date) training face model with batch $BATCH" >> $LOG
"${AWAKE[@]}" .venv/bin/python train_face.py --data data/face_aligned_v9 --batch $BATCH --resume 2>&1 \
  | grep -E --line-buffered "aligned face pairs|iter [0-9]*000/|Traceback|Error" >> $LOG
echo "$(date) phase 4 finished" >> $LOG
