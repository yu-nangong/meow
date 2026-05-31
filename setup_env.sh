#!/usr/bin/env bash
# MEOW worktree setup: data symlink + venv + Python deps.
# Safe to re-run on resume (skips steps that are already OK).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Worktree: 6 levels up to moew/; seed dir: 4 levels (script still works from either).
if [[ -d "$(pwd)/data" ]] && [[ -n "$(ls -A "$(pwd)/data"/*.h5 2>/dev/null || true)" ]]; then
  : # data/ already resolves to H5 files
else
  for rel in ../../../../../../data ../../../../data; do
    target="$(cd "$ROOT" && cd "$rel" 2>/dev/null && pwd)" || continue
    if [[ -d "$target" ]] && ls "$target"/*.h5 &>/dev/null; then
      ln -sfn "$rel" data
      break
    fi
  done
fi

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

_venv_ok() {
  [[ -x "$PY" ]] && "$PY" -c "import sys" &>/dev/null
}

if ! _venv_ok; then
  rm -rf "$VENV"
  # Inherit conda/base packages when pip/network is unavailable.
  python3 -m venv --system-site-packages "$VENV"
fi

if "$PY" -c "import tables, sklearn, pandas, numpy" &>/dev/null; then
  echo "[setup_env] MEOW deps OK ($( "$PY" -c 'import numpy; print(numpy.__version__)' ))"
else
  echo "[setup_env] Installing tables scikit-learn pandas numpy ..."
  export PIP_DEFAULT_TIMEOUT=20
  "$PIP" install --retries 1 tables scikit-learn pandas numpy
fi

if ! "$PY" -c "import tables, sklearn, pandas, numpy" &>/dev/null; then
  echo "[setup_env] ERROR: MEOW deps still missing (network/DNS?). Try:"
  echo "  conda install -y -c conda-forge numpy pandas pytables scikit-learn"
  echo "  cd $(pwd) && rm -rf .venv && bash setup_env.sh"
  exit 1
fi

if ! "$PY" -c "import torch" &>/dev/null; then
  echo "[setup_env] Installing torch (optional) ..."
  "$PIP" install --retries 1 -q torch \
    || echo "[setup_env] torch skipped (optional; Ridge path works without it)"
fi
