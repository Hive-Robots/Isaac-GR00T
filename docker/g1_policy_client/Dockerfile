FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_NO_BANNER=1
ENV CYCLONEDDS_HOME=/opt/cyclonedds/install
ENV CMAKE_PREFIX_PATH=/opt/cyclonedds/install
ENV LD_LIBRARY_PATH=${MAMBA_ROOT_PREFIX}/envs/gr00t/lib:/opt/cyclonedds/install/lib:${LD_LIBRARY_PATH}
ENV WORKSPACE=/workspace/gr00t
ENV PYTHONPATH=/workspace/gr00t
ENV PATH=${MAMBA_ROOT_PREFIX}/envs/gr00t/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libusb-1.0-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds /opt/cyclonedds/src \
    && cmake -S /opt/cyclonedds/src -B /opt/cyclonedds/src/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds/install \
    && cmake --build /opt/cyclonedds/src/build -j"$(nproc)" --target install \
    && rm -rf /opt/cyclonedds/src

RUN arch="$(dpkg --print-architecture)" \
    && case "${arch}" in \
        amd64) mm_arch="64" ;; \
        arm64) mm_arch="aarch64" ;; \
        *) echo "Unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac \
    && curl -Ls "https://micro.mamba.pm/api/micromamba/linux-${mm_arch}/latest" \
    | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && micromamba create -y -n gr00t -c conda-forge python=3.10 pinocchio casadi pip \
    && micromamba clean --all --yes

COPY docker/g1_policy_client/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && grep -Ev '^(casadi|pin)==.*$' /tmp/requirements.txt > /tmp/requirements.pip.txt \
    && python -m pip install -r /tmp/requirements.pip.txt \
    && python -m pip install cyclonedds==0.10.2 \
    && python -c "import pinocchio as pin; from pinocchio import casadi as cpin; print('pinocchio version:', pin.__version__); print('pinocchio.casadi module:', cpin.__name__)"

COPY external_dependencies/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python /opt/unitree_sdk2_python

RUN python -m pip install --no-deps -e /opt/unitree_sdk2_python \
    && rm -rf /tmp/requirements.txt /tmp/requirements.pip.txt

WORKDIR ${WORKSPACE}

# Copy only the codepaths needed by eval_g1/eval_g1_loop.
COPY gr00t ${WORKSPACE}/gr00t
COPY examples/g1_XRtele ${WORKSPACE}/examples/g1_XRtele
COPY examples/g1_XRtele_wristcams ${WORKSPACE}/examples/g1_XRtele_wristcams

CMD ["/bin/bash"]
