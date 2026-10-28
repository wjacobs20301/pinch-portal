#!/usr/bin/env bash
# Phase 5: anime eyes back on the face model. ~15 h. Resumes after interruptions (runs as filter-face5.service).
#  1. teacher with the expression lock on mouth/jaw/nose ONLY (eyes/brows blanked from the landmark guide)
#  2. real colours (anime eyes kept) + brighter lighting + alignment
#  3. sharper face model: start from face_v9, stronger critic (0.08) and line loss (2.0)
cd "$(dirname "$0")"
LOG=logs_phase5.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AWAKE=(systemd-inhibit --what=sleep:idle --who=filter-app --why="Training anime face model")
echo "$(date) phase 5: mouth-only expression lock teacher" >> $LOG
"${AWAKE[@]}" .venv/bin/python teacher_face.py --src data/raw_faces_full --dst data/faces_v10 --face 1.1 --mouth-only 2>&1 \
  | grep -E --line-buffered "photos to process|00/|failed|Traceback|Error:" >> $LOG
echo "$(date) colours + lighting + alignment" >> $LOG
.venv/bin/python align_faces.py data/faces_v10 --out data/face_aligned_v10 2>&1 | grep -E "aligned|Traceback|Error" >> $LOG
BATCH=""
for b in 16 12 8; do
  if timeout 900 .venv/bin/python train_face.py --name preflight --batch $b --iters 20 --gan-start 0 \
       --data data/face_aligned_v10 > /tmp/preflight_face.log 2>&1; then BATCH=$b; break; fi
  echo "$(date) batch $b failed: $(grep -m1 -E 'Error|error' /tmp/preflight_face.log)" >> $LOG
done
rm -f checkpoints/preflight*
[ -z "$BATCH" ] && { echo "$(date) phase 5: no batch size fits, stopping" >> $LOG; exit 1; }
echo "$(date) training face model v10 with batch $BATCH" >> $LOG
"${AWAKE[@]}" .venv/bin/python train_face.py --name face_v10 --data data/face_aligned_v10 --init checkpoints/face_v9.pt \
  --gan-weight 0.08 --line 2.0 --gan-start 0 --iters 40000 --batch $BATCH --resume 2>&1 \
  | grep -E --line-buffered "aligned face pairs|initialised|iter [0-9]*000/|Traceback|Error" >> $LOG
echo "$(date) phase 5 finished" >> $LOG
