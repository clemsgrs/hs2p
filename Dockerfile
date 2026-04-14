FROM ubuntu:22.04

ARG USER_UID=1001
ARG USER_GID=1001

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam
ENV PATH="/home/user/.local/bin:${PATH}"

# Create user and I/O dirs
RUN groupadd --gid ${USER_GID} user \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user \
    && mkdir /input /output \
    && chown user:user /input /output

WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtiff-dev \
    zlib1g-dev \
    curl \
    cmake \
    vim screen \
    zip unzip \
    git \
    openssh-server \
    build-essential \
    ninja-build \
    python3-pip python3-dev python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# libjpeg-turbo 3.x (required by PyTurboJPEG>=2)
ARG LIBJPEG_TURBO_VERSION=3.1.0
RUN curl -fsSL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz \
      | tar xz -C /tmp \
    && cd /tmp/libjpeg-turbo-${LIBJPEG_TURBO_VERSION} \
    && cmake -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local . \
    && make -j"$(nproc)" && make install \
    && ldconfig \
    && rm -rf /tmp/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}

# ASAP
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN set -eux; \
    apt-get update; \
    curl -L "${ASAP_URL}" -o /tmp/ASAP.deb; \
    apt-get install -y --no-install-recommends /tmp/ASAP.deb; \
    SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"); \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth"; \
    rm -f /tmp/ASAP.deb; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Python deps & pyproject-based package install
RUN python -m pip install --upgrade pip "setuptools>=61" wheel pip-tools \
    && rm -rf /root/.cache/pip

ARG GIT_MODEL_DEPENDENCIES="git+https://github.com/facebookresearch/sam2.git"

COPY --chown=user:user pyproject.toml README.md LICENSE /opt/app/
COPY --chown=user:user hs2p /opt/app/hs2p
RUN python -m pip install \
      --no-cache-dir \
      --no-color \
      --no-build-isolation \
      "/opt/app[all,sam2]" \
      ${GIT_MODEL_DEPENDENCIES} \
      && rm -rf /root/.cache/pip

COPY --chown=user:user . /opt/app/

USER user
WORKDIR /opt/app
