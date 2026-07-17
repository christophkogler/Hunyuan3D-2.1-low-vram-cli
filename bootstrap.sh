#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export MAX_JOBS="${MAX_JOBS:-2}"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -)"
fi
value="all"
install_command=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) value="${2:-}"; shift 2 ;;
    --install-command) install_command=true; shift ;;
    *) echo "usage: ./bootstrap.sh --profile {shape|texture|all} [--install-command]" >&2; exit 2 ;;
  esac
done
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it first: https://docs.astral.sh/uv/" >&2
  exit 2
fi
case "$value" in
  shape) extras="" ;;
  all|texture) extras="--extra texture" ;;
  *) echo "usage: ./bootstrap.sh --profile {shape|texture|all}" >&2; exit 2 ;;
esac
uv sync --locked --python 3.11 $extras
if [[ "$value" != "shape" ]]; then
  (cd hy3dpaint/custom_rasterizer && uv run --no-sync python setup.py build_ext --inplace)
  suffix="$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
  c++ -O3 -Wall -shared -std=c++11 -fPIC $(uv run --no-sync python -m pybind11 --includes) hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpp -o "hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor${suffix}"
fi
if [[ "$install_command" == true ]]; then
  "$(dirname "$0")/install-command.sh"
fi
echo "Bootstrap complete. Run: .venv/bin/hunyuan3d doctor"
