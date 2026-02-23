#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-g1-policy-client:latest}"
BUILD_CTX="$(mktemp -d)"

cleanup() {
    rm -rf "${BUILD_CTX}"
}
trap cleanup EXIT

mkdir -p "${BUILD_CTX}/docker/g1_policy_client"
mkdir -p "${BUILD_CTX}/external_dependencies/GR00T-WholeBodyControl/external_dependencies"
mkdir -p "${BUILD_CTX}/examples"

cp "${SCRIPT_DIR}/Dockerfile" "${BUILD_CTX}/docker/g1_policy_client/Dockerfile"
cp "${SCRIPT_DIR}/requirements.txt" "${BUILD_CTX}/docker/g1_policy_client/requirements.txt"
cp -r "${REPO_ROOT}/gr00t" "${BUILD_CTX}/gr00t"
cp -r "${REPO_ROOT}/examples/g1_XRtele" "${BUILD_CTX}/examples/g1_XRtele"
cp -r "${REPO_ROOT}/examples/g1_XRtele_wristcams" "${BUILD_CTX}/examples/g1_XRtele_wristcams"
cp -r \
    "${REPO_ROOT}/external_dependencies/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python" \
    "${BUILD_CTX}/external_dependencies/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python"

docker build \
    --network host \
    -f "${BUILD_CTX}/docker/g1_policy_client/Dockerfile" \
    -t "${IMAGE_NAME}" \
    "$@" \
    "${BUILD_CTX}"

echo "Built image: ${IMAGE_NAME}"
