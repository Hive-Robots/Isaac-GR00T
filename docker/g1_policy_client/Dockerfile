FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CYCLONEDDS_HOME=/opt/cyclonedds/install
ENV CMAKE_PREFIX_PATH=/opt/cyclonedds/install
ENV LD_LIBRARY_PATH=/opt/cyclonedds/install/lib:${LD_LIBRARY_PATH}
ENV WORKSPACE=/workspace/gr00t
ENV PYTHONPATH=/workspace/gr00t

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
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

COPY docker/g1_policy_client/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements.txt \
    && python -m pip install cyclonedds==0.10.2

COPY external_dependencies/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python /tmp/unitree_sdk2_python

RUN python -m pip install --no-deps /tmp/unitree_sdk2_python \
    && rm -rf /tmp/unitree_sdk2_python /tmp/requirements.txt

WORKDIR ${WORKSPACE}

# Copy only the codepaths needed by eval_g1/eval_g1_loop.
COPY gr00t ${WORKSPACE}/gr00t
COPY examples/g1_XRtele ${WORKSPACE}/examples/g1_XRtele
COPY examples/g1_XRtele_wristcams ${WORKSPACE}/examples/g1_XRtele_wristcams

CMD ["/bin/bash"]
