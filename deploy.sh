#!/usr/bin/env bash
# Push the app (page + current anime models) to the machine that serves it.
# Settings come from deploy.env (not committed), e.g.:
#   DEPLOY_SSH="ssh"  DEPLOY_HOST="myserver"  DEPLOY_DIR="filter-app/web"  DEPLOY_URL="https://example.com"
set -euo pipefail
cd "$(dirname "$0")"
[ -f deploy.env ] && source deploy.env
: "${DEPLOY_HOST:?set DEPLOY_HOST (and DEPLOY_DIR) in deploy.env}"
DEPLOY_SSH="${DEPLOY_SSH:-ssh}"; DEPLOY_DIR="${DEPLOY_DIR:-filter-app/web}"
rsync -az -e "$DEPLOY_SSH" index.html "$DEPLOY_HOST:$DEPLOY_DIR/"
rsync -az -e "$DEPLOY_SSH" models/anime_student.onnx "$DEPLOY_HOST:$DEPLOY_DIR/models/"
# the dedicated face model is optional: the app skips the face step when it's missing
if [ -f models/anime_face.onnx ]; then rsync -az -e "$DEPLOY_SSH" models/anime_face.onnx "$DEPLOY_HOST:$DEPLOY_DIR/models/"; fi
echo "deployed → ${DEPLOY_URL:-$DEPLOY_HOST} (refresh the page)"
