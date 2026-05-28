#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-venv}"
INSTALL_TORCH="${INSTALL_TORCH:-auto}"

usage() {
  cat <<'EOF'
Usage: scripts/setup_environment.sh [--with-torch|--without-torch]

Creates ./venv, installs requirements.txt, and optionally installs CPU
PyTorch/TorchVision using the README's Raspberry Pi friendly indexes.

Options:
  --with-torch     Install torch and torchvision after requirements.
  --without-torch  Skip torch and torchvision installation.

Environment:
  PYTHON_BIN       Python executable to use. Default: python3
  VENV_DIR         Virtual environment directory. Default: venv
  INSTALL_TORCH    auto, 1, true, yes, 0, false, no. Default: auto
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-torch)
      INSTALL_TORCH="true"
      shift
      ;;
    --without-torch)
      INSTALL_TORCH="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
  VENV_PYTHON="$VENV_DIR/bin/python"
  ACTIVATE_COMMAND="source $VENV_DIR/bin/activate"
elif [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
  VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
  ACTIVATE_COMMAND="$VENV_DIR\\Scripts\\Activate.ps1"
else
  echo "Could not find Python inside virtual environment: $VENV_DIR" >&2
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt

should_install_torch=false
case "${INSTALL_TORCH,,}" in
  1|true|yes)
    should_install_torch=true
    ;;
  0|false|no)
    should_install_torch=false
    ;;
  auto)
    if ! "$VENV_PYTHON" -c "import torch, torchvision" >/dev/null 2>&1; then
      should_install_torch=true
    fi
    ;;
  *)
    echo "Invalid INSTALL_TORCH value: $INSTALL_TORCH" >&2
    exit 2
    ;;
esac

if [[ "$should_install_torch" == true ]]; then
  "$VENV_PYTHON" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://www.piwheels.org/simple \
    torch

  "$VENV_PYTHON" -m pip install torchvision --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://www.piwheels.org/simple
fi

cat <<EOF

Environment setup complete.

Activate it with:
  $ACTIVATE_COMMAND

For manual module runs, set:
  export PYTHONPATH="$PROJECT_ROOT"
EOF
