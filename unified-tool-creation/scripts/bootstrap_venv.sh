#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch_root="$(realpath -m "${RECO_SCRATCH_ROOT:-/tmp/reco-atlas}")"
venv_dir="$(realpath -m "${RECO_VENV_PATH:-${scratch_root}/venv}")"
python_bin="${RECO_PYTHON:-python3.11}"

case "${venv_dir}" in
  "${scratch_root}"/*) ;;
  *)
    echo "Refusing to create a bulk environment outside ${scratch_root}: ${venv_dir}" >&2
    exit 2
    ;;
esac

export PIP_CACHE_DIR="${scratch_root}/pip-cache"
export TMPDIR="${scratch_root}/tmp"
export HF_HOME="${scratch_root}/hf"
export HF_DATASETS_CACHE="${scratch_root}/hf/datasets"
export HUGGINGFACE_HUB_CACHE="${scratch_root}/hf/hub"

mkdir -p \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}" \
  "${HF_DATASETS_CACHE}" \
  "${HUGGINGFACE_HUB_CACHE}"

"${python_bin}" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        f"This project environment requires Python 3.11+; got {sys.version.split()[0]}"
    )
PY

# --clear safely replaces a partial or older environment at this exact scratch path.
"${python_bin}" -m venv --clear "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install -r "${project_dir}/requirements.txt"

if [[ "${RECO_KEEP_PIP_CACHE:-0}" != "1" ]]; then
  "${venv_dir}/bin/python" -m pip cache purge
fi

echo
echo "Environment ready. In each new shell run:"
echo "  cd ${project_dir}"
echo "  source ${venv_dir}/bin/activate"
echo
echo "Use this interpreter in the IDE:"
echo "  ${venv_dir}/bin/python"
