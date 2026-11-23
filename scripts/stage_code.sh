#!/usr/bin/env bash
# Upload the reviewed worktree as a content-addressed package without requiring a TPU.
# Normally called by launch_training.sh after the selected local YAML has been validated.
set -euo pipefail

H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$H/.." && pwd)"
if [ "$#" -ne 0 ]; then
    echo "FATAL: usage: stage_code.sh (normally called by launch_training.sh)" >&2
    exit 2
fi
# shellcheck source=_deployment_env.sh
source "$H/_deployment_env.sh"

BUCKET=${BUCKET:-$GCS_BUCKET}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

tar czf "$TMP/pkg.tgz" --exclude=__pycache__ --exclude='*.pyc' \
    -C "$REPO" ./README.md ./pyproject.toml ./uv.lock ./.python-version \
    ./LICENSE ./LICENSES ./THIRD_PARTY_NOTICES.md \
    ./src ./scripts/_bootstrap.sh ./configs/training/small.example.yaml \
    ./configs/training/full.example.yaml "./$TRAINING_CONFIG"
HASH=$(md5sum "$TMP/pkg.tgz" | cut -c1-12)
PKG="$TMP/fastwam_jax_${HASH}.tgz"
mv "$TMP/pkg.tgz" "$PKG"
gcloud storage cp "$PKG" "gs://$BUCKET/$CODE_PREFIX/fastwam_jax_${HASH}.tgz"
printf '%s\n' "$HASH" > "${FASTWAM_STAGED_HASH_FILE:-$REPO/.last_sync_hash}"
echo "STAGED hash=$HASH uri=gs://$BUCKET/$CODE_PREFIX/fastwam_jax_${HASH}.tgz"
