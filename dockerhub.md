# iMontage on RunPod (DockerHub image)

This container is built for RunPod Pods/Serverless. It bootstraps iMontage at runtime so you always get the latest code from the fork.

## Image
- `gemneye/imontage:latest` (each commit also has a `gemneye/imontage:<git-sha>` tag)

## What the container does at start
1) Installs Python deps (Torch 2.7.0/cu128, torchvision 0.22.0, torchaudio 2.7.0, flash-attn 3.0.0.post1).  
2) Clones the repo into `/workspace/iMontage` if it is not present.  
3) `pip install -e .` to bring in project deps.  
4) Launches the Gradio UI (`gradio_app.py`) on port `7860`.

## Required downloads (models)
Before running inference you must download checkpoints per the project README into `ckpts/` (e.g., `ckpts/hyvideo_ckpts/...`, `ckpts/iMontage_ckpts/...`). You can bake this into a volume or perform the download inside the container startup if desired.

## Environment overrides
- `TORCH_VERSION`, `TORCHVISION_VERSION`, `TORCHAUDIO_VERSION`, `TORCH_INDEX_URL` – pick different torch wheels if needed.
- `FLASH_ATTN_VERSION` (flash-attn required; avoid setting `INSTALL_FLASH_ATTN=0` unless you know it’s safe to skip).
- `REPO_URL`, `REPO_DIR` – point to a different fork/location.
- `MODEL_BASE`, `DIT_WEIGHT` – override checkpoint paths used by the Gradio app.

## Running on RunPod Pods
1) Use `gemneye/imontage:latest` as the image.  
2) Expose port `7860` (container port) and map to your chosen proxy port in the template.  
3) Ensure a volume or startup command downloads the required `ckpts` before first inference.  
4) The default command runs the Gradio UI. To run the CLI instead, override the container command with something like:  
   `["python3.12", "fastvideo/sample/sample_imontage.py", "--prompt", "assets/prompt.json", ...]`

## Running locally
```bash
docker run --gpus all -it -p 7860:7860 --ipc=host gemneye/imontage:latest
# Gradio UI will be at http://localhost:7860
```
