#!/usr/bin/env bash
# setup_env.sh — create/activate a Python 3.10 virtual environment and install deps.
#
# Usage (WSL / Linux — these 42 projects are Linux-tested):
#   source ./setup_env.sh      # build .venv (py3.10), install deps, and ACTIVATE it
#   ./setup_env.sh             # build + install only (then: source .venv/bin/activate)
#   PYTHON=/path/to/python3.10 source ./setup_env.sh   # use a specific interpreter
#
# Python 3.10 is required because tensorflow / pygame / mne / scikit-learn ship
# prebuilt wheels for it (the campus / WSL default 3.14 has none).

# Were we sourced? (so activation can persist in the caller's shell)
_srcd=0
[ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ] && _srcd=1

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# 1) Locate a Python 3.10 interpreter.
_py=""
if [ -n "${PYTHON:-}" ]; then
  _py="$PYTHON"
elif command -v python3.10 >/dev/null 2>&1; then
  _py="python3.10"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys;exit(0 if sys.version_info[:2]==(3,10) else 1)' 2>/dev/null; then
  _py="python3"
elif command -v pyenv >/dev/null 2>&1; then
  _v="$(pyenv versions --bare 2>/dev/null | grep -E '^3\.10\.' | tail -1)"
  [ -n "$_v" ] && _py="$(pyenv root)/versions/$_v/bin/python"
fi

if [ -z "$_py" ]; then
  echo "setup_env: no Python 3.10 interpreter found." >&2
  echo "  Ubuntu/WSL : sudo apt update && sudo apt install -y python3.10 python3.10-venv" >&2
  echo "               (older releases first: sudo add-apt-repository ppa:deadsnakes/ppa)" >&2
  echo "  pyenv      : pyenv install 3.10.14" >&2
  echo "  or point at one: PYTHON=/full/path/to/python3.10 source ./setup_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

echo "setup_env: using $("$_py" --version 2>&1)"

# 2) Create the venv if needed.
if [ ! -d "$_here/.venv" ]; then
  "$_py" -m venv "$_here/.venv" || {
    echo "setup_env: venv creation failed — is the python3.10-venv package installed?" >&2
    return 1 2>/dev/null || exit 1
  }
  echo "setup_env: created .venv"
fi

# 3) Activate, then install dependencies.
# shellcheck source=/dev/null
source "$_here/.venv/bin/activate"
python -m pip install --upgrade pip >/dev/null
if [ -s "$_here/requirements.txt" ]; then
  python -m pip install -r "$_here/requirements.txt"
fi

echo ""
if [ "$_srcd" = 1 ]; then
  echo "Environment ready and ACTIVE — $(python --version 2>&1).  Leave it with: deactivate"
else
  echo "Environment ready.  Activate it with:  source .venv/bin/activate"
fi
