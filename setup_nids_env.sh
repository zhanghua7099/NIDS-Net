#!/usr/bin/env bash
#
# One-click setup for the NIDS-Net demo (GroundingDINO + SAM + DINOv2).
# Creates a conda env "nids" (Python 3.9), installs the full CUDA 11.8 /
# PyTorch 2.2.1 / detectron2 / GroundingDINO / MobileSAM stack, downloads
# the required checkpoints, and (optionally) runs demo_eval_gdino_FFA.py
# as a smoke test.
#
# Tested on: Ubuntu, NVIDIA GPU with driver >= 470, conda already installed.
#
# Usage:
#   ./setup_nids_env.sh            # full setup + run the demo at the end
#   ./setup_nids_env.sh --no-demo  # full setup, skip running the demo
#
# NOTE: no "-u" (nounset) here. conda's own scripts (conda.sh, and package
# activate.d/deactivate.d hooks such as binutils_linux-64's ADDR2LINE or
# gxx_linux-64's CONDA_BACKUP_CXX) are not nounset-safe, and "conda install"
# itself internally activates/deactivates the target env to run those hooks
# -- not just our explicit "conda activate" call -- so nounset can't be
# safely scoped to just a few lines.
set -eo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENV_NAME="nids"
PY_VERSION="3.9.18"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DEMO=1
[[ "${1:-}" == "--no-demo" ]] && RUN_DEMO=0

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

GCC_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/x86_64-conda-linux-gnu-gcc"
GXX_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/x86_64-conda-linux-gnu-g++"
CUDA_HOME_DIR="$CONDA_BASE/envs/$ENV_NAME"

log() { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\n\033[1;33m!! $*\033[0m" >&2; }

# Retry a flaky (usually network-bound) command with exponential backoff.
# Machines differ a lot in network reliability to PyPI/conda/GitHub/Box
# mirrors, and a single dropped connection shouldn't take down the whole
# multi-step install.
retry() {
  local attempt=1 max=5 delay=10
  until "$@"; do
    local status=$?
    if (( attempt >= max )); then
      warn "Command failed after $attempt attempts (exit $status): $*"
      return "$status"
    fi
    warn "Attempt $attempt/$max failed (exit $status), retrying in ${delay}s: $*"
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

# On any uncaught failure, say where and remind the user the script is safe
# to just re-run: every step below is gated on "is this already done?" so a
# re-run resumes near the failure point instead of starting over.
trap 'warn "setup_nids_env.sh failed at line $LINENO: $BASH_COMMAND"; \
      warn "Re-run ./setup_nids_env.sh to resume -- already-completed steps (env, packages, checkpoints) are detected and skipped."' ERR

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------
command -v conda >/dev/null || { echo "conda not found on PATH"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found; a CUDA GPU is required"; exit 1; }

# Auto-detect the GPU's compute capability(ies) so extensions build for the
# right arch(es). Dedup'd + semicolon-joined so this also works on
# multi-GPU boxes with mixed generations (e.g. "7.5;8.6").
TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
  | tr -d ' ' | sort -u | paste -sd ';' -)"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
log "Detected TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# ---------------------------------------------------------------------------
# 1. Create the conda env (Python only — pip deps come after torch below)
# ---------------------------------------------------------------------------
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  log "Conda env '$ENV_NAME' already exists, reusing it"
else
  log "Creating conda env '$ENV_NAME' (python=$PY_VERSION)"
  retry conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi
conda activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# 1b. Pin numpy<2 for the lifetime of this env, not just this script.
#     torch/detectron2/pycocotools wheels here are built against the numpy
#     1.x ABI; numpy 2.x makes torch.from_numpy() blow up at runtime with
#     "RuntimeError: Numpy is not available". Packages like opencv-python
#     declare "numpy>=2" and will happily pull in numpy 2.x the moment any
#     pip install resolves them -- including GroundingDINO/MobileSAM's own
#     "pip install" call inside setup.py (step 6 below), and any package you
#     pip install by hand later in this env. A pip constraints file (via
#     PIP_CONSTRAINT) applies to all of those uniformly, since subprocess
#     pip calls inherit the shell env. Persisting it into the env's
#     activate.d hook makes it stick for future terminal sessions too, so
#     this can't regress by a stray "pip install X" after setup is done.
# ---------------------------------------------------------------------------
ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"
CONSTRAINTS_FILE="$ENV_DIR/numpy-constraint.txt"
echo "numpy<2" > "$CONSTRAINTS_FILE"
export PIP_CONSTRAINT="$CONSTRAINTS_FILE"

mkdir -p "$ENV_DIR/etc/conda/activate.d" "$ENV_DIR/etc/conda/deactivate.d"
cat > "$ENV_DIR/etc/conda/activate.d/pin-numpy.sh" <<EOF
export PIP_CONSTRAINT="$CONSTRAINTS_FILE"
EOF
cat > "$ENV_DIR/etc/conda/deactivate.d/pin-numpy.sh" <<'EOF'
unset PIP_CONSTRAINT
EOF

# ---------------------------------------------------------------------------
# 2. PyTorch 2.2.1 + CUDA 11.8 (install this FIRST so nothing else drags in
#    a mismatched torch build from pip later)
# ---------------------------------------------------------------------------
log "Installing PyTorch 2.2.1 / torchvision / torchaudio (CUDA 11.8)"
retry conda install -y -n "$ENV_NAME" pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  pytorch-cuda=11.8 -c pytorch -c nvidia

log "Installing full CUDA 11.8 toolkit (nvcc + dev headers: cudart, cusparse, cublas, ...)"
retry conda install -y -n "$ENV_NAME" -c nvidia/label/cuda-11.8.0 cuda-toolkit

log "Installing gcc/g++ 11 (nvcc 11.8 rejects newer system gcc) + ninja"
retry conda install -y -n "$ENV_NAME" -c conda-forge gxx_linux-64=11 gcc_linux-64=11 ninja

log "Installing xformers (must match torch==2.2.1 + cu11.8 exactly)"
retry conda install -y -n "$ENV_NAME" xformers -c xformers

# mkl>=2025 removes a symbol (iJIT_NotifyEvent) that this torch build needs.
log "Pinning mkl to a version compatible with torch 2.2.1"
retry pip install "mkl==2024.0.0"

# ---------------------------------------------------------------------------
# 3. Remaining pip dependencies (torch is already pinned, so these won't
#    upgrade it)
# ---------------------------------------------------------------------------
log "Installing remaining Python dependencies"
retry pip install \
  omegaconf torchmetrics==0.10.3 fvcore iopath opencv-python pycocotools \
  matplotlib onnxruntime onnx scipy hydra-colorlog hydra-core gdown \
  pytorch-lightning pandas ruamel.yaml pyrender wandb distinctipy chardet \
  requests tqdm ftfy regex absl-py
# opencv-python's newest build declares numpy>=2, but PIP_CONSTRAINT (set in
# step 1b) forces pip to resolve an opencv-python build compatible with
# numpy<2 instead, so this never drags numpy 2.x in.

# ---------------------------------------------------------------------------
# 4. detectron2 (build from source; must NOT use pip's build isolation or
#    it can't see the torch we just installed)
# ---------------------------------------------------------------------------
if python -c "import detectron2" >/dev/null 2>&1; then
  log "detectron2 already importable, skipping build"
else
  log "Building detectron2 from source (this takes a few minutes)"
  D2_SRC="$(mktemp -d)/detectron2"
  retry git clone --quiet https://github.com/facebookresearch/detectron2.git "$D2_SRC"
  ( cd "$D2_SRC" && \
    CC="$GCC_BIN" CXX="$GXX_BIN" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    CUDA_HOME="$CUDA_HOME_DIR" \
    pip install --no-build-isolation . )
fi

# ---------------------------------------------------------------------------
# 5. SAM (segment-anything) + supervision
# ---------------------------------------------------------------------------
log "Installing segment-anything + supervision==0.20.0"
retry pip install 'git+https://github.com/facebookresearch/segment-anything.git' supervision==0.20.0

# ---------------------------------------------------------------------------
# 6. RoboKit + GroundingDINO + MobileSAM (setup.py also downloads their
#    checkpoints into ckpts/gdino and ckpts/mobilesam)
# ---------------------------------------------------------------------------
cd "$REPO_DIR"
if python -c "from groundingdino.models import build_model; import mobile_sam" >/dev/null 2>&1; then
  log "GroundingDINO / MobileSAM already installed, skipping"
else
  log "Running setup.py install (RoboKit -> pulls GroundingDINO + MobileSAM + weights)"
  CC="$GCC_BIN" CXX="$GXX_BIN" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    CUDA_HOME="$CUDA_HOME_DIR" python setup.py install
fi

# Download <url> to <dest> atomically: retry with resume (-C -) into a
# ".part" file and only mv it into place on full success, so a dropped
# connection never leaves behind a truncated file that a later "-f dest"
# existence check would mistake for a completed download.
download() {
  local url=$1 dest=$2 tmp="${2}.part"
  retry curl -fL --retry 5 --retry-delay 5 --retry-connrefused -C - -o "$tmp" "$url"
  mv "$tmp" "$dest"
}

# ---------------------------------------------------------------------------
# 7. SAM ViT-H checkpoint
# ---------------------------------------------------------------------------
mkdir -p "$REPO_DIR/ckpts/sam_weights"
SAM_CKPT="$REPO_DIR/ckpts/sam_weights/sam_vit_h_4b8939.pth"
if [[ -f "$SAM_CKPT" ]]; then
  log "SAM ViT-H checkpoint already present, skipping download"
else
  log "Downloading SAM ViT-H checkpoint (~2.4GB)"
  download https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth "$SAM_CKPT"
fi

# ---------------------------------------------------------------------------
# 8. Template object feature embeddings (High-resolution demo dataset)
# ---------------------------------------------------------------------------
mkdir -p "$REPO_DIR/obj_FFA"
FEAT_JSON="$REPO_DIR/obj_FFA/object_features_vitl14_reg.json"
if [[ -f "$FEAT_JSON" ]]; then
  log "Template embeddings already present, skipping download"
else
  log "Downloading template object feature embeddings (~50MB)"
  # NOTE: this Box link 404s on a HEAD request; must GET with -L (redirects).
  download "https://utdallas.box.com/shared/static/50a8q7i5hc33rovgyavoiw0utuduno39" "$FEAT_JSON"
  # Box occasionally serves an HTML error/login page with a 200 status
  # instead of the file; catch that here rather than at import time later.
  if ! python -c "import json; json.load(open('$FEAT_JSON'))" >/dev/null 2>&1; then
    rm -f "$FEAT_JSON"
    echo "Downloaded file is not valid JSON (Box likely served an error page). Re-run the script to retry." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 9. DINOv2 (torch.hub): the "main" branch on GitHub now uses Python 3.10+
#    union-type syntax ("float | None") which breaks under Python 3.9.
#    Prime the hub cache, then patch it with `from __future__ import
#    annotations` so the same source works on 3.9.
# ---------------------------------------------------------------------------
log "Priming + patching DINOv2 torch.hub cache for Python 3.9 compatibility"
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', skip_validation=True)" \
  >/dev/null 2>&1 || true

HUB_DIR="$(python -c 'import torch; print(torch.hub.get_dir())')/facebookresearch_dinov2_main"
for f in "$HUB_DIR/dinov2/layers/attention.py" "$HUB_DIR/dinov2/layers/block.py"; do
  if [[ -f "$f" ]] && ! grep -q "from __future__ import annotations" "$f"; then
    sed -i '1i from __future__ import annotations' "$f"
  fi
done

# Now this should succeed and cache the model weights too.
retry python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', skip_validation=True); print('DINOv2 OK')"

# ---------------------------------------------------------------------------
# 10. Final verification
# ---------------------------------------------------------------------------
log "Verifying full stack"
python - <<'PY'
import torch, torchvision, detectron2, numpy
from groundingdino.models import build_model
import mobile_sam

# A numpy 2.x regression here means torch.from_numpy() will raise
# "RuntimeError: Numpy is not available" at inference time -- fail loudly
# now instead of discovering it later mid-inference.
major = int(numpy.__version__.split(".")[0])
assert major < 2, (
    f"numpy {numpy.__version__} is installed but torch/detectron2 need numpy<2. "
    "PIP_CONSTRAINT should have prevented this -- check "
    "$CONDA_PREFIX/numpy-constraint.txt and re-run: pip install 'numpy<2'"
)

print("torch       ", torch.__version__, "cuda_available=", torch.cuda.is_available())
print("torchvision ", torchvision.__version__)
print("detectron2  ", detectron2.__version__)
print("numpy       ", numpy.__version__)
print("All imports OK")
PY

# ---------------------------------------------------------------------------
# 11. Run the demo
# ---------------------------------------------------------------------------
if [[ "$RUN_DEMO" -eq 1 ]]; then
  log "Running demo_eval_gdino_FFA.py"
  cd "$REPO_DIR"
  MPLBACKEND=Agg python demo_eval_gdino_FFA.py
  log "Demo finished. Visualizations written to:"
  echo "  $REPO_DIR/exps/demo0501_448_mask/predictions/test_002_gt.jpg"
  echo "  $REPO_DIR/exps/demo0501_448_mask/predictions/test_002_pred_SAM+DINOv2.jpg"
else
  log "Setup complete. Run the demo manually with:"
  echo "  conda activate $ENV_NAME && cd $REPO_DIR && MPLBACKEND=Agg python demo_eval_gdino_FFA.py"
fi
