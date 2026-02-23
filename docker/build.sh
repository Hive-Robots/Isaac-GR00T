#!/bin/bash

set -x

image_name="gr00t-dev"
default_base_image="nvcr.io/nvidia/pytorch:25.04-py3"
arm64_base_image="nvcr.io/nvidia/pytorch:25.08-py3"

export DOCKER_BUILDKIT=1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Copy gr00t directory to src/gr00t
mkdir -p $DIR/src
rm -rf /tmp/gr00t

echo $DIR

cp -r $DIR/../ /tmp/gr00t
cp -r /tmp/gr00t $DIR/src/

export DOCKER_BUILDKIT=1

# Filter out --fix flag and other script-specific flags before passing to docker
docker_args=()
for arg in "$@"; do
    case $arg in
        --fix)
            # Skip --fix flag as it's not a valid docker build flag
            ;;
        *)
            docker_args+=("$arg")
            ;;
    esac
done

host_arch="$(uname -m)"
case "$host_arch" in
    x86_64|amd64)
        detected_platform="linux/amd64"
        detected_base_image="$default_base_image"
        ;;
    aarch64|arm64)
        detected_platform="linux/arm64"
        detected_base_image="$arm64_base_image"
        ;;
    *)
        detected_platform="linux/amd64"
        detected_base_image="$default_base_image"
        echo "Unsupported host architecture '$host_arch'. Falling back to $detected_platform and $detected_base_image."
        ;;
esac

docker_platform="${DOCKER_PLATFORM:-$detected_platform}"
base_image="${BASE_IMAGE:-$detected_base_image}"

echo "Host architecture: $host_arch"
echo "Using platform: $docker_platform"
echo "Using base image: $base_image"

docker build "${docker_args[@]}" \
    --build-arg BASE_IMAGE="$base_image" \
    --platform "$docker_platform" \
    --network host \
    -t "$image_name" "$DIR" \
    && echo Image $image_name BUILT SUCCESSFULLY

rm -rf "$DIR/src/"
