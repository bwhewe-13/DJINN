
#!/usr/bin/env bash
# setup_envs.sh — Clone the TF repo and create both DJINN virtual environments
#
# Usage (run from the root of your PyTorch fork):
#   bash setup_envs.sh
#
# The PyTorch venv installs from the current directory (your fork).
# The TF repo is cloned into ./repos/DJINN-tf automatically.
# To use a different TF checkout, override TF_REPO:
#   TF_REPO=/path/to/llnl-djinn bash setup_envs.sh

set -e

TF_URL="https://github.com/LLNL/DJINN.git"

REPOS_DIR="./repos"
TF_REPO="${TF_REPO:-$REPOS_DIR/DJINN-tf}"
PT_REPO="${PT_REPO:-$(pwd)}"
VENV_DIR="./venvs"


clone_or_update() {
    local url="$1" dest="$2" name="$3"
    if [ -d "$dest/.git" ]; then
        echo "  '$dest' already exists — pulling latest..."
        git -C "$dest" pull --ff-only
    else
        echo "  Cloning $name from $url → $dest"
        git clone "$url" "$dest"
    fi
}

# Clone repos and create venvs
echo ""
echo "=== Cloning repositories ==="
mkdir -p "$REPOS_DIR"
clone_or_update "$TF_URL" "$TF_REPO" "LLNL TensorFlow DJINN"
echo "  PyTorch fork: using current repo at $PT_REPO"

mkdir -p "$VENV_DIR"

# Tensorflow
echo ""
echo "=== Creating TensorFlow DJINN environment ==="
python3 -m venv "$VENV_DIR/tf-djinn"
source "$VENV_DIR/tf-djinn/bin/activate"

pip install --upgrade pip --quiet
pip install tensorflow scikit-learn scipy numpy matplotlib pytest --quiet
pip install -e "$TF_REPO" --quiet

echo "Installed packages (tf-djinn):"
pip list | grep -E "tensorflow|torch|djinn|sklearn|scipy|numpy"
deactivate

# PyTorch
echo ""
echo "=== Creating PyTorch DJINN environment ==="
python3 -m venv "$VENV_DIR/pt-djinn"
source "$VENV_DIR/pt-djinn/bin/activate"

pip install --upgrade pip --quiet
pip install torch scikit-learn scipy numpy matplotlib pytest --quiet
pip install -e "$PT_REPO" --quiet

echo "Installed packages (pt-djinn):"
pip list | grep -E "tensorflow|torch|djinn|sklearn|scipy|numpy"
deactivate


echo ""
echo "Both environments created in $VENV_DIR/"
echo ""
echo "Next steps:"
echo "  1. Run unit tests in each env:"
echo "       source $VENV_DIR/tf-djinn/bin/activate && pytest test_unit.py -v && deactivate"
echo "       source $VENV_DIR/pt-djinn/bin/activate && pytest test_unit.py -v && deactivate"
echo ""
echo "  2. Collect benchmark results:"
echo "       source $VENV_DIR/tf-djinn/bin/activate"
echo "       python run_and_collect.py --impl tf --out results_tf.json"
echo "       deactivate"
echo ""
echo "       source $VENV_DIR/pt-djinn/bin/activate"
echo "       python run_and_collect.py --impl pt --out results_pt.json"
echo "       deactivate"
echo ""
echo "  3. Compare:"
echo "       python compare_results.py --tf results_tf.json --pt results_pt.json --plot"
