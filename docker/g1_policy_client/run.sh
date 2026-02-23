#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-g1-policy-client:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-g1-policy-client}"

if [ "$#" -eq 0 ]; then
    set -- /bin/bash
fi

docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    -v "${REPO_ROOT}:/workspace/gr00t" \
    -w /workspace/gr00t \
    "${IMAGE_NAME}" \
    "$@"
