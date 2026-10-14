#!/usr/bin/env bash
set -euo pipefail
# Install first with uv sync --frozen --extra agent --extra grammar.
# Do not implicitly change packages/download GBs merely to inspect --dry-run.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${RECOATLAS_PYTHON:-$repo_dir/.venv/bin/python}" -m recoatlas.sweep "$@"
