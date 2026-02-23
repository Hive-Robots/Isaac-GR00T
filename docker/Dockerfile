ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
FROM ${BASE_IMAGE}

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

RUN apt-get update && apt-get install -y \
    build-essential yasm cmake libtool git pkg-config \
    libass-dev libfreetype6-dev libvorbis-dev \
    autoconf automake texinfo tmux ffmpeg

RUN pip install --upgrade pip setuptools
RUN pip install uv
RUN if [ -f /etc/pip/constraint.txt ]; then sed -i s/dill==0.3.9/dill==0.3.8/g /etc/pip/constraint.txt; fi
RUN if [ -f /etc/pip/constraint.txt ]; then sed -i s/scipy==1.15.2/scipy==1.15.3/g /etc/pip/constraint.txt; fi
RUN set -eux; \
    pip install imageio h5py boto3 transformers[torch] deepspeed timm peft diffusers wandb tianshou dm_tree openai albumentations==1.4.18; \
    arch="$(uname -m)"; \
    if [ "$arch" = "x86_64" ] || [ "$arch" = "amd64" ]; then \
        pip install decord==0.6.0 torchcodec==0.4.0; \
    else \
        echo "Skipping decord/torchcodec on $arch (no stable PyPI wheel support)."; \
    fi
RUN mkdir -p /opt
RUN cd /opt && git clone https://github.com/facebookresearch/pytorch3d.git
RUN cd /opt/pytorch3d && pip install --no-build-isolation .

COPY src/gr00t/ /workspace/gr00t/
RUN cd /workspace/gr00t && uv sync && uv pip install -e .

RUN apt-get update && apt-get install -y libegl1
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat >/usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat >/usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.2.140"
    }
}
EOF
