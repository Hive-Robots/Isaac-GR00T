# Docker Setup for NVIDIA Isaac GR00T

Docker configuration for building and running a containerized GR00T environment with all dependencies pre-installed. The image (`gr00t-dev`) is based on NVIDIA's PyTorch container and includes CUDA support, Python dependencies, PyTorch3D, and the GR00T codebase.

## Prerequisites

- Docker (version 20.10+) and [perform post-installation setup](https://docs.docker.com/engine/install/linux-postinstall/) to verify that you can run docker commands without sudo. 
- NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- NVIDIA GPU with compatible drivers
- Bash shell
- Sufficient disk space (several GB)

## Building the Docker Image

Make sure you are using a bash environment:

```bash
bash build.sh
```

The build script auto-detects host architecture and selects platform/base image defaults:
- `x86_64` host: `linux/amd64` + `nvcr.io/nvidia/pytorch:25.04-py3`
- `aarch64` host (Jetson Thor): `linux/arm64` + `nvcr.io/nvidia/pytorch:25.08-py3`

You can override either value:

```bash
DOCKER_PLATFORM=linux/arm64 BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 bash build.sh
```

The build process installs all dependencies and sets up the GR00T codebase at `/workspace/gr00t/`.

On ARM64 (Jetson Thor), the build skips `decord==0.6.0` and `torchcodec==0.4.0` because stable PyPI wheels are not consistently available for `aarch64`.
On ARM64 (Jetson Thor), `flash-attn` is excluded by dependency markers because the `uv` Python 3.10 environment typically resolves CPU-only `torch` and does not provide `nvcc/CUDA_HOME` for compiling flash-attn.
PyTorch3D is installed with `--no-build-isolation` so its build step can use the preinstalled `torch` from the base image.

## Running the Container

**Interactive shell (uses code baked into image):**
```bash
docker run -it --rm --gpus all gr00t-dev /bin/bash
```

**Interactive shell on Jetson Thor (NVIDIA runtime mode):**
```bash
docker run -it --rm --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    gr00t-dev /bin/bash
```

**Development mode (mounts local codebase for live editing):**
```bash
docker run -it --rm --gpus all \
    -v $(pwd)/..:/workspace/gr00t \
    gr00t-dev /bin/bash
```
**Development mode on Jetson Thor (NVIDIA runtime mode):**
```bash
docker run -it --rm --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    -v $(pwd)/..:/workspace/gr00t \
    gr00t-dev /bin/bash
```
**Run this from the `docker/` directory. Changes to your local GR00T code will be immediately reflected inside the container.**


## Troubleshooting

**GPU not detected:**
- Verify NVIDIA Container Toolkit: `nvidia-container-toolkit --version`
- Restart Docker: `sudo systemctl restart docker`
- Test GPU access: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`

**Permission errors:**
- Use `sudo` with Docker commands, or add your user to the `docker` group: `sudo usermod -aG docker $USER`

**Build failures:**
- Check disk space: `df -h`
- Clean Docker: `docker system prune -a`
- Rebuild: `sudo bash build.sh --no-cache`
