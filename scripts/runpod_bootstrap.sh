#!/usr/bin/env bash
# Bootstraps iMontage inside a fresh RunPod container. Installs Python deps,
# clones the repo if needed, ensures models are present (downloads if missing),
# then execs the provided command (default: Gradio app).
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
REPO_URL="${REPO_URL:-https://github.com/sruckh/iMontage.git}"
REPO_DIR="${REPO_DIR:-/workspace/iMontage}"
MARKER_FILE="${MARKER_FILE:-/workspace/.imontage_bootstrap_done}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
# Optional direct wheel URL to avoid index misses; default matches torch 2.7.0/cu12.
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}" # flash-attn is required; set to 0 only if you know you don't need it

CKPTS_ROOT="${CKPTS_ROOT:-${REPO_DIR}/ckpts}"
HYVIDEO_DIR="${HYVIDEO_DIR:-${CKPTS_ROOT}/hyvideo_ckpts}"
IMONTAGE_DIR="${IMONTAGE_DIR:-${CKPTS_ROOT}/iMontage_ckpts}"
LLAVA_DIR="${LLAVA_DIR:-${CKPTS_ROOT}/llava-llama-3-8b-v1_1-transformers}"
TEXT_ENCODER_DIR="${TEXT_ENCODER_DIR:-${HYVIDEO_DIR}/text_encoder}"
TEXT_ENCODER_2_DIR="${TEXT_ENCODER_2_DIR:-${HYVIDEO_DIR}/text_encoder_2}"
IMONTAGE_WEIGHT_PATH="${IMONTAGE_WEIGHT_PATH:-${IMONTAGE_DIR}/diffusion_pytorch_model.safetensors}"
HYVIDEO_MARKER="${HYVIDEO_MARKER:-${HYVIDEO_DIR}/.download_complete}"
LLAVA_MARKER="${LLAVA_MARKER:-${LLAVA_DIR}/.download_complete}"
TEXT_ENCODER_MARKER="${TEXT_ENCODER_MARKER:-${TEXT_ENCODER_DIR}/.processed}"
TEXT_ENCODER_2_MARKER="${TEXT_ENCODER_2_MARKER:-${TEXT_ENCODER_2_DIR}/.download_complete}"
IMONTAGE_MARKER="${IMONTAGE_MARKER:-${IMONTAGE_DIR}/.download_complete}"

log() {
    echo "[runpod-bootstrap] $*"
}

if [[ ! -x "$(command -v "${PYTHON_BIN}")" ]]; then
    log "ERROR: ${PYTHON_BIN} not found; ensure python3.12 is installed."
    exit 1
fi

# Clone repo if missing
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    log "Cloning iMontage repo into ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

maybe_install_deps() {
    if [[ -f "${MARKER_FILE}" ]]; then
        log "Dependencies already installed (marker present at ${MARKER_FILE}). Remove it to reinstall."
        return
    fi

    log "Skipping pip upgrade (using system pip to avoid Debian uninstall errors)"

    log "Installing PyTorch ${TORCH_VERSION} (cu128) and friends from ${TORCH_INDEX_URL}"
    "${PYTHON_BIN}" -m pip install --break-system-packages \
        torch=="${TORCH_VERSION}" \
        torchvision=="${TORCHVISION_VERSION}" \
        torchaudio=="${TORCHAUDIO_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"

    if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
        if [[ -n "${FLASH_ATTN_WHL}" ]]; then
            log "Installing flash-attn from wheel (required): ${FLASH_ATTN_WHL}"
            if ! "${PYTHON_BIN}" -m pip install "${FLASH_ATTN_WHL}" --no-build-isolation --break-system-packages; then
                log "ERROR: flash-attn install failed from wheel ${FLASH_ATTN_WHL}"
                exit 1
            fi
        else
            log "Installing flash-attn ${FLASH_ATTN_VERSION} (required)."
            if ! "${PYTHON_BIN}" -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation --break-system-packages; then
                log "ERROR: flash-attn install failed for ${FLASH_ATTN_VERSION}. Set FLASH_ATTN_VERSION/FLASH_ATTN_WHL to a compatible release for your torch/CUDA."
                exit 1
            fi
        fi
    else
        log "WARNING: Skipping flash-attn install (INSTALL_FLASH_ATTN=0) even though it is required by README."
    fi

    log "Installing iMontage Python package and dependencies (-e .)"
    "${PYTHON_BIN}" -m pip install -e . --break-system-packages

    log "Bootstrap complete."
    touch "${MARKER_FILE}"
}

install_st_attn() {
    if "${PYTHON_BIN}" -c "import st_attn" >/dev/null 2>&1; then
        log "st_attn already installed."
        return
    fi

    log "Installing st_attn (Sliding Tile Attention)..."
    # Clone FastVideo temporarily to build the extension
    local tmp_dir
    tmp_dir=$(mktemp -d)
    
    log "Cloning FastVideo repo to ${tmp_dir}..."
    git clone https://github.com/hao-ai-lab/FastVideo.git "${tmp_dir}/FastVideo"
    
    cd "${tmp_dir}/FastVideo/csrc/sliding_tile_attention"
    log "Building st_attn..."
    if "${PYTHON_BIN}" -m pip install . --break-system-packages; then
        log "Successfully installed st_attn."
    else
        log "WARNING: Failed to install st_attn. Inference will be slower."
    fi
    
    # Cleanup
    cd "${REPO_DIR}"
    rm -rf "${tmp_dir}"
}

ensure_hf_cli() {
    if command -v hf >/dev/null 2>&1; then
        echo "hf"
        return
    fi
    if command -v huggingface-cli >/dev/null 2>&1; then
        # Prefer hf, but fall back if only huggingface-cli exists
        echo "huggingface-cli"
        return
    fi
    log "Installing huggingface_hub[cli] to obtain hf"
    "${PYTHON_BIN}" -m pip install --upgrade "huggingface_hub[cli]" --break-system-packages
    if command -v hf >/dev/null 2>&1; then
        echo "hf"
        return
    fi
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli"
        return
    fi
    echo ""
}

dir_has_content() {
    [[ -d "$1" ]] && find "$1" -mindepth 1 -print -quit >/dev/null 2>&1
}

download_if_missing() {
    local label="$1"
    local model="$2"
    local target="$3"
    local hf_cmd="$4"
    local required_file="$5" # optional file to verify presence
    local marker_file="$6"    # optional marker to note success

    if [[ -n "${marker_file}" && -f "${marker_file}" ]]; then
        log "Found ${label} marker at ${marker_file}; skipping download."
        return
    fi
    if [[ -n "${required_file}" && -f "${required_file}" ]]; then
        log "Found ${label} required file at ${required_file}; skipping download."
        return
    fi

    log "Downloading ${label} (${model}) to ${target}"
    mkdir -p "${target}"
    if ! "${hf_cmd}" download "${model}" --local-dir "${target}"; then
        log "ERROR: failed to download ${label} (${model})"
        exit 1
    fi

    if [[ -n "${required_file}" && ! -f "${required_file}" ]]; then
        log "ERROR: download completed but required file missing: ${required_file}"
        exit 1
    fi
    if [[ -n "${marker_file}" ]]; then
        touch "${marker_file}"
    fi
    log "Downloaded ${label} to ${target}"
}

maybe_download_models() {
    HF_CMD=$(ensure_hf_cli)
    if [[ -z "${HF_CMD}" ]]; then
        log "ERROR: huggingface CLI not available; cannot auto-download models."
        return
    fi

    export HF_TOKEN="${HF_TOKEN:-}"
    mkdir -p "${HYVIDEO_DIR}" "${IMONTAGE_DIR}" "${TEXT_ENCODER_DIR}" "${TEXT_ENCODER_2_DIR}"

    download_if_missing "HunyuanVideo-I2V" "tencent/HunyuanVideo-I2V" "${HYVIDEO_DIR}" "${HF_CMD}" "" "${HYVIDEO_MARKER}"
    download_if_missing "LLaVA tokenizer" "xtuner/llava-llama-3-8b-v1_1-transformers" "${LLAVA_DIR}" "${HF_CMD}" "" "${LLAVA_MARKER}"
    if dir_has_content "${LLAVA_DIR}" && ! dir_has_content "${TEXT_ENCODER_DIR}"; then
        log "Running tokenizer preprocess to ${TEXT_ENCODER_DIR}"
        if ! "${PYTHON_BIN}" fastvideo/models/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
            --input_dir "${LLAVA_DIR}" \
            --output_dir "${TEXT_ENCODER_DIR}"; then
            log "ERROR: tokenizer preprocess failed"
            exit 1
        fi
        touch "${TEXT_ENCODER_MARKER}"
    fi
    download_if_missing "CLIP text_encoder_2" "openai/clip-vit-large-patch14" "${TEXT_ENCODER_2_DIR}" "${HF_CMD}" "" "${TEXT_ENCODER_2_MARKER}"
    download_if_missing "iMontage weights" "Kr1sJ/iMontage" "${IMONTAGE_DIR}" "${HF_CMD}" "${IMONTAGE_WEIGHT_PATH}" "${IMONTAGE_MARKER}"

    ensure_vae_link
    ensure_text_encoder_link
    ensure_text_encoder_2_link
}

ensure_vae_link() {
    local target="${REPO_DIR}/data/hunyuan/hunyuan-video-i2v-720p/vae"
    if [[ -d "${target}" ]]; then
        log "Found VAE at ${target}"
        return
    fi
    local candidate
    candidate=$(find "${HYVIDEO_DIR}" -maxdepth 3 -type d -name "vae" | head -n 1 || true)
    if [[ -z "${candidate}" ]]; then
        log "ERROR: VAE directory not found under ${HYVIDEO_DIR}"
        exit 1
    fi
    mkdir -p "$(dirname "${target}")"
    ln -s "${candidate}" "${target}"
    log "Linked VAE from ${candidate} to ${target}"
}

ensure_text_encoder_link() {
    local target="${REPO_DIR}/data/hunyuan/text_encoder"
    if [[ -d "${target}" ]]; then
        log "Found text encoder at ${target}"
        return
    fi
    if [[ -d "${LLAVA_DIR}" ]]; then
        mkdir -p "$(dirname "${target}")"
        ln -s "${LLAVA_DIR}" "${target}"
        log "Linked text encoder from ${LLAVA_DIR} to ${target}"
        return
    fi
    log "ERROR: text encoder directory not found at ${LLAVA_DIR}"
    exit 1
}

ensure_text_encoder_2_link() {
    local target="${REPO_DIR}/data/hunyuan/text_encoder_2"
    if [[ -d "${target}" ]]; then
        log "Found text encoder 2 at ${target}"
        return
    fi
    if [[ -d "${TEXT_ENCODER_2_DIR}" ]]; then
        mkdir -p "$(dirname "${target}")"
        ln -s "${TEXT_ENCODER_2_DIR}" "${target}"
        log "Linked text encoder 2 from ${TEXT_ENCODER_2_DIR} to ${target}"
        return
    fi
    log "ERROR: text encoder 2 directory not found at ${TEXT_ENCODER_2_DIR}"
    exit 1
}

maybe_install_deps

install_st_attn

maybe_download_models

if "${PYTHON_BIN}" -c "import st_attn" >/dev/null 2>&1; then
    log "SUCCESS: st_attn is successfully imported and available."
else
    log "WARNING: st_attn could NOT be imported. Performance may be impacted."
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${REPO_DIR}/scripts:${PATH}"

log "Starting command: $*"
exec "$@"