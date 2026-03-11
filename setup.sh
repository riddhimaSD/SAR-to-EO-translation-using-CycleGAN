#!/usr/bin/env bash
# =============================================================================
# setup.sh
# Environment setup script for SAR-to-EO CycleGAN project.
#
# Usage:
#   bash setup.sh               # Creates venv, installs deps, verifies imports
#   bash setup.sh --no-venv     # Installs deps into the current environment
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
PYTHON=${PYTHON:-python3}
VENV_DIR="venv"
USE_VENV=true

# Parse arguments
for arg in "$@"; do
    case $arg in
        --no-venv) USE_VENV=false ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# -----------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------
echo "============================================================"
echo "  SAR-to-EO CycleGAN -- Environment Setup"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------
# 1. Python version check
# -----------------------------------------------------------------------
echo "[1/5] Checking Python version..."
PYTHON_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_MAJOR=3
REQUIRED_MINOR=9

MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$MAJOR" -lt "$REQUIRED_MAJOR" ] || { [ "$MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$MINOR" -lt "$REQUIRED_MINOR" ]; }; then
    echo "ERROR: Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ required. Found $PYTHON_VERSION."
    exit 1
fi
echo "  Python $PYTHON_VERSION detected. OK"
echo ""

# -----------------------------------------------------------------------
# 2. Create virtual environment (optional)
# -----------------------------------------------------------------------
if [ "$USE_VENV" = true ]; then
    echo "[2/5] Creating virtual environment in ./$VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"

    # Activate (Linux/macOS or Windows Git Bash)
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        source "$VENV_DIR/Scripts/activate"
    else
        echo "WARNING: Could not activate venv automatically."
        echo "         Run manually: source $VENV_DIR/bin/activate"
    fi
    echo "  Virtual environment created and activated."
else
    echo "[2/5] Skipping virtual environment (--no-venv)."
fi
echo ""

# -----------------------------------------------------------------------
# 3. Upgrade pip
# -----------------------------------------------------------------------
echo "[3/5] Upgrading pip..."
$PYTHON -m pip install --upgrade pip --quiet
echo "  pip upgraded."
echo ""

# -----------------------------------------------------------------------
# 4. Install dependencies
# -----------------------------------------------------------------------
echo "[4/5] Installing dependencies from requirements.txt..."
$PYTHON -m pip install -r requirements.txt
echo "  Dependencies installed."
echo ""

# -----------------------------------------------------------------------
# 5. Verify imports
# -----------------------------------------------------------------------
echo "[5/5] Verifying key imports..."

$PYTHON - <<'EOF'
import importlib, sys

modules = [
    ("torch",           "PyTorch"),
    ("torchvision",     "Torchvision"),
    ("rasterio",        "Rasterio"),
    ("pytorch_msssim",  "pytorch-msssim"),
    ("matplotlib",      "Matplotlib"),
    ("tqdm",            "tqdm"),
    ("yaml",            "PyYAML"),
]

failed = []
for mod, label in modules:
    try:
        importlib.import_module(mod)
        print(f"  {label:20s} OK")
    except ImportError as e:
        print(f"  {label:20s} FAILED -- {e}")
        failed.append(label)

if failed:
    print(f"\nSetup incomplete. Failed imports: {failed}")
    sys.exit(1)
else:
    print("\nAll imports verified.")
EOF

# -----------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup complete."
echo ""
echo "  Train (Part A - RGB):"
echo "    python train.py --config configs/config_part_a.yaml"
echo ""
echo "  Evaluate:"
echo "    python evaluate.py --config configs/config_part_a.yaml"
echo "============================================================"
