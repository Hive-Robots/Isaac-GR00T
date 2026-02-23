#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-g1-policy-client:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-g1-policy-client}"
UNITREE_SDK2_SRC="/workspace/gr00t/external_dependencies/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python"
CONTAINER_PYTHONPATH="${CONTAINER_PYTHONPATH:-/workspace/gr00t:${UNITREE_SDK2_SRC}}"

if [ "$#" -eq 0 ]; then
    set -- /bin/bash
fi

docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    -e PYTHONPATH="${CONTAINER_PYTHONPATH}" \
    -v "${REPO_ROOT}:/workspace/gr00t" \
    -w /workspace/gr00t \
    "${IMAGE_NAME}" \
    "$@"
