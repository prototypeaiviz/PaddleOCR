#!/usr/bin/env bash
set -euo pipefail

# This script creates and activates a virtual environment workflow.
# Run it from the project root.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "Environment created. Activate it with: source .venv/bin/activate"
