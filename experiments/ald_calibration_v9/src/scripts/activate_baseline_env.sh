#!/usr/bin/env bash
# Usage: source /absolute/path/to/evoproto/scripts/activate_baseline_env.sh
_EVOPROTO_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
_EVOPROTO_RUNTIME="${_EVOPROTO_PROJECT_ROOT}/.runtime"
export CONDA_PKGS_DIRS="${_EVOPROTO_RUNTIME}/pkgs"
export CONDA_ENVS_PATH="${_EVOPROTO_RUNTIME}/envs"
export CONDA_ALWAYS_COPY=true
export CONDA_AUTO_UPDATE_CONDA=false
export CONDA_NUMBER_CHANNEL_NOTICES=0
export PIP_CACHE_DIR="${_EVOPROTO_RUNTIME}/cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export XDG_CACHE_HOME="${_EVOPROTO_RUNTIME}/cache/xdg"
export TMPDIR="${_EVOPROTO_RUNTIME}/tmp"
export TORCH_HOME="${_EVOPROTO_RUNTIME}/cache/torch"
export CUDA_CACHE_PATH="${_EVOPROTO_RUNTIME}/cache/cuda"
export TRITON_CACHE_DIR="${_EVOPROTO_RUNTIME}/cache/triton"
export MPLCONFIGDIR="${_EVOPROTO_RUNTIME}/cache/matplotlib"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${_EVOPROTO_RUNTIME}/env"
unset _EVOPROTO_PROJECT_ROOT _EVOPROTO_RUNTIME
