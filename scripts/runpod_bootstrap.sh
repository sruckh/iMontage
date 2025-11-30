#!/usr/bin/env bash
# Bootstraps iMontage inside a fresh RunPod container. Installs Python deps,
# clones the repo if needed, ensures models are present (downloads if missing),
# then execs the provided command (default: Gradio app).
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_URL="${REPO_URL:-https://github.com/sruckh/iMontage.git}"
REPO_DIR="${REPO_DIR:-/workspace/iMontage}"
MARKER_FILE="${MARKER_FILE:-/workspace/.imontage_bootstrap_done}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-3.0.0.post1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}" # flash-attn is required; set to 0 only if you know you don't need it

CKPTS_ROOT="${CKPTS_ROOT:-${REPO_DIR}/ckpts}"
HYVIDEO_DIR="${HYVIDEO_DIR:-${CKPTS_ROOT}/hyvideo_ckpts}"
IMONTAGE_DIR="${IMONTAGE_DIR:-${CKPTS_ROOT}/iMontage_ckpts}"
LLAVA_DIR="${LLAVA_DIR:-${CKPTS_ROOT}/llava-llama-3-8b-v1_1-transformers}"
TEXT_ENCODER_DIR="${TEXT_ENCODER_DIR:-${HYVIDEO_DIR}/text_encoder}"
TEXT_ENCODER_2_DIR="${TEXT_ENCODER_2_DIR:-${HYVIDEO_DIR}/text_encoder_2}"

log() {
    echo "[runpod-bootstrap] $*"
}

if [[ ! -x "$(command -v "${PYTHON_BIN}")" ]]; then
    log "ERROR: ${PYTHON_BIN} not found; ensure python3 is installed."
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

    log "Upgrading pip"
    "${PYTHON_BIN}" -m pip install --upgrade pip

    log "Installing PyTorch ${TORCH_VERSION} (cu128) and friends from ${TORCH_INDEX_URL}"
    "${PYTHON_BIN}" -m pip install \
        torch=="${TORCH_VERSION}" \
        torchvision=="${TORCHVISION_VERSION}" \
        torchaudio=="${TORCHAUDIO_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"

    if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
        log "Installing flash-attn ${FLASH_ATTN_VERSION} (required)."
        if ! "${PYTHON_BIN}" -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation; then
            log "ERROR: flash-attn install failed. This image uses the CUDA devel base for build tooling; if it still fails, check CUDA compatibility or adjust FLASH_ATTN_VERSION."
            exit 1
        fi
    else
        log "WARNING: Skipping flash-attn install (INSTALL_FLASH_ATTN=0) even though it is required by README."
    fi

    log "Installing iMontage Python package and dependencies (-e .)"
    "${PYTHON_BIN}" -m pip install -e .

    log "Bootstrap complete."
    touch "${MARKER_FILE}"
}

ensure_hf_cli() {
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli"
        return
    fi
    if command -v hf >/dev/null 2>&1; then
        echo "hf"
        return
    fi
    log "Installing huggingface_hub[cli] to obtain huggingface-cli"
    "${PYTHON_BIN}" -m pip install --upgrade "huggingface_hub[cli]"
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli"
        return
    fi
    if command -v hf >/dev/null 2>&1; then
        echo "hf"
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
    if dir_has_content "${target}"; then
        log "Found ${label} at ${target}; skipping download."
        return
    fi
    log "Downloading ${label} (${model}) to ${target}"
    mkdir -p "${target}"
    if ! "${hf_cmd}" download "${model}" --local-dir "${target}"; then
        log "WARNING: failed to download ${label} (${model})"
    fi
}

maybe_download_models() {
    HF_CMD=$(ensure_hf_cli)
    if [[ -z "${HF_CMD}" ]]; then
        log "ERROR: huggingface CLI not available; cannot auto-download models."
        return
    fi

    export HF_TOKEN="${HF_TOKEN:-}"
    mkdir -p "${HYVIDEO_DIR}" "${IMONTAGE_DIR}" "${TEXT_ENCODER_DIR}" "${TEXT_ENCODER_2_DIR}"

    download_if_missing "HunyuanVideo-I2V" "tencent/HunyuanVideo-I2V" "${HYVIDEO_DIR}" "${HF_CMD}"
    download_if_missing "LLaVA tokenizer" "xtuner/llava-llama-3-8b-v1_1-transformers" "${LLAVA_DIR}" "${HF_CMD}"
    if dir_has_content "${LLAVA_DIR}" && ! dir_has_content "${TEXT_ENCODER_DIR}"; then
        log "Running tokenizer preprocess to ${TEXT_ENCODER_DIR}"
        "${PYTHON_BIN}" fastvideo/models/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
            --input_dir "${LLAVA_DIR}" \
            --output_dir "${TEXT_ENCODER_DIR}" || log "WARNING: tokenizer preprocess failed"
    fi
    download_if_missing "CLIP text_encoder_2" "openai/clip-vit-large-patch14" "${TEXT_ENCODER_2_DIR}" "${HF_CMD}"
    download_if_missing "iMontage weights" "Kr1sJ/iMontage" "${IMONTAGE_DIR}" "${HF_CMD}"
}

maybe_install_deps
maybe_download_models

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PATH="${REPO_DIR}/scripts:${PATH}"

log "Starting command: $*"
exec "$@"
