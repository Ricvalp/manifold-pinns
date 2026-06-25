#!/usr/bin/env bash
# Source this file on Snellius before uv sync or experiment commands:
#   source env_snellius.sh
#
# Override SNELLIUS_SCRATCH or MANIFOLD_PINNS_STORAGE_ROOT if your project has
# a dedicated scratch/project allocation.

_mp_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${MANIFOLD_PINNS_STORAGE_ROOT:-}" ]; then
  if [ -n "${SNELLIUS_SCRATCH:-}" ]; then
    export MANIFOLD_PINNS_STORAGE_ROOT="${SNELLIUS_SCRATCH}/manifold-pinns"
  elif [ -d "/scratch-shared/${USER}" ] && [ -w "/scratch-shared/${USER}" ]; then
    export MANIFOLD_PINNS_STORAGE_ROOT="/scratch-shared/${USER}/manifold-pinns"
  elif [ -n "${TMPDIR:-}" ] && [ -d "${TMPDIR}" ] && [ -w "${TMPDIR}" ]; then
    export MANIFOLD_PINNS_STORAGE_ROOT="${TMPDIR}/manifold-pinns"
  else
    export MANIFOLD_PINNS_STORAGE_ROOT="${HOME}/scratch/manifold-pinns"
  fi
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export MANIFOLD_PINNS_WANDB_USE="${MANIFOLD_PINNS_WANDB_USE:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

source "${_mp_repo_root}/env.sh"
