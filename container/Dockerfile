# SNAX-FORGE development container (D118): one image for snax-forge,
# snax-gvsoc and the programs built from snax_cluster.
#
# Holds tools only: GVSoC's build tools and Python packages, snax_cluster's
# RISC-V toolchain, pixi and zsh. No repository is in the image: mount the
# folder that holds snax-forge, snax-gvsoc and snax_cluster at its own path, and
# build there, so edits and build output stay on the host. Each repo keeps its
# own environment: `pixi install` in snax-forge writes snax-forge/.pixi/.
#
# Moved here from snax-gvsoc (b029ea4) in SG0.
#
# Two stages, as in snax_cluster's util/container/Dockerfile: pixi installs the
# locked toolchain in the first, and only the environment is copied into the
# second, at the same path, since a conda environment has its path built in.
#
# Build (from the snax-forge root):
#   docker build -t ghcr.io/kuleuven-micas/snax-forge:main container
# Run (from the folder that holds the repos):
#   podman run --rm -it -v "$PWD":"$PWD" -w "$PWD" ghcr.io/kuleuven-micas/snax-forge:main

########################################################################
# Stage 1: the RISC-V toolchain, installed with pixi
########################################################################
FROM ubuntu:24.04 AS toolchain

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# pixi 0.81.0 here: it wrote toolchain/pixi.lock (lock format 7), which the
# 0.41.4 in the final image cannot read.
ARG PIXI_VERSION=0.81.0
ARG PIXI_SHA256=7aa3ec39aecceff9062fa2ed4d42cbaa0bdc25ddea727d048e061cf188d434f6
RUN curl -fsSL --retry 5 -o /tmp/pixi.tar.gz \
        "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
    && echo "${PIXI_SHA256}  /tmp/pixi.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/pixi.tar.gz -C /usr/local/bin pixi \
    && rm /tmp/pixi.tar.gz

# clang, lld, llvm-tools, clang-format 21.1.6 and bender 0.28.2: the exact
# builds snax_cluster's pixi.lock holds (see toolchain/pixi.toml).
COPY toolchain/pixi.toml toolchain/pixi.lock /opt/toolchain/
RUN cd /opt/toolchain && pixi install --locked

########################################################################
# Stage 2: the image
########################################################################
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Host compiler and build tools for GVSoC, plus zsh.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ccache \
        git \
        ca-certificates \
        curl \
        xz-utils \
        python3 \
        python3-venv \
        zlib1g-dev \
        zsh \
    && rm -rf /var/lib/apt/lists/*

COPY --from=toolchain /opt/toolchain/.pixi/envs/default /opt/toolchain/.pixi/envs/default

# Python packages in a venv, so nothing fights the system Python.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# zsh with oh-my-zsh and the autosuggestions plugin, as snax_cluster's image
# had them (53efdeb), but at fixed versions and with oh-my-zsh's update
# check off, so a rebuild gives the same shell.
ARG OMZ_COMMIT=083c339e0622bfc144b66ebf1a2e8dbe7d2ada0d
ARG ZSH_AUTOSUGGESTIONS_TAG=v0.7.1
RUN git init -q /root/.oh-my-zsh \
    && git -C /root/.oh-my-zsh fetch -q --depth 1 https://github.com/ohmyzsh/ohmyzsh.git "$OMZ_COMMIT" \
    && git -C /root/.oh-my-zsh -c advice.detachedHead=false checkout -q FETCH_HEAD \
    && sed "s|^# zstyle ':omz:update' mode disabled|zstyle ':omz:update' mode disabled|" \
        /root/.oh-my-zsh/templates/zshrc.zsh-template > /root/.zshrc \
    && git clone -q --depth 1 --branch "$ZSH_AUTOSUGGESTIONS_TAG" \
        https://github.com/zsh-users/zsh-autosuggestions.git \
        /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions \
    && echo "source ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh" >> /root/.zshrc \
    && chsh -s /usr/bin/zsh

# pixi for the repos' own environments: 0.41.4, the version snax-forge's CI pins
# and its pixi.lock was written with, so `pixi install` behaves as in CI.
ARG PIXI_PROJECT_VERSION=0.41.4
ARG PIXI_PROJECT_SHA256=ce029857d08983fbfdb90d43a99c82035c51cd0921e90187313f134de2fb3ec1
RUN curl -fsSL --retry 5 -o /tmp/pixi.tar.gz \
        "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_PROJECT_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
    && echo "${PIXI_PROJECT_SHA256}  /tmp/pixi.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/pixi.tar.gz -C /usr/local/bin pixi \
    && rm /tmp/pixi.tar.gz

# The toolchain goes last on PATH, so nothing in it can shadow the host tools
# GVSoC builds with. Build GVSoC outside `pixi shell`: snax-forge's environment
# sets CC and CXX to conda's gcc.
ENV PATH=/opt/venv/bin:$PATH:/opt/toolchain/.pixi/envs/default/bin

# The repos are bind-mounted and owned by the host user.
RUN git config --system --add safe.directory '*'

CMD ["zsh"]
