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
    vim screen \
    zip unzip \
    git \
    openssh-server \
    build-essential \
    ninja-build \
    python3-pip python3-dev python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

# Python deps & app
RUN python -m pip install --upgrade pip setuptools pip-tools \
    && rm -rf /root/.cache/pip

COPY --chown=user:user requirements.in /opt/app/requirements.in
RUN python -m pip install \
      --no-cache-dir \
      --no-color \
      --requirement /opt/app/requirements.in \
    && rm -rf /root/.cache/pip

COPY --chown=user:user . /opt/app/
RUN python -m pip install /opt/app

USER user
WORKDIR /opt/app
