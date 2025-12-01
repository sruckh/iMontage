"""Simple Gradio interface for running iMontage inference on RunPod.

This wraps the existing `fastvideo/sample/sample_imontage.py` script and launches
it via `torchrun --nproc_per_node=1` so that all upstream distributed logic
remains intact.

The app assumes model checkpoints are already present (see README download
instructions). If they are missing, the UI will surface a descriptive error
before attempting inference.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = os.environ.get("MODEL_BASE", str(REPO_ROOT / "ckpts" / "hyvideo_ckpts"))
DEFAULT_DIT_WEIGHT = os.environ.get(
    "DIT_WEIGHT", str(REPO_ROOT / "ckpts" / "iMontage_ckpts" / "diffusion_pytorch_model.safetensors")
)


def _check_required_files() -> Tuple[bool, str]:
    missing: List[str] = []
    if not Path(DEFAULT_MODEL_ROOT).exists():
        missing.append(f"model_path ({DEFAULT_MODEL_ROOT})")
    if not Path(DEFAULT_DIT_WEIGHT).exists():
        missing.append(f"dit-weight ({DEFAULT_DIT_WEIGHT})")
    if missing:
        return False, (
            "Missing required checkpoints: "
            + ", ".join(missing)
            + ". Download ckpts per README before running inference."
        )
    return True, ""


def _write_prompt_json(
    task_type: str,
    prompt_text: str,
    image_paths: List[Path],
    height: int,
    width: int,
    output_num: int,
) -> Path:
    payload = {
        "0": {
            "task_type": task_type,
            "prompts": prompt_text,
            "images": [str(p) for p in image_paths],
            "height": height,
            "width": width,
            "output_num": output_num,
        }
    }
    tmp_json = Path(tempfile.mkdtemp()) / "prompt.json"
    tmp_json.write_text(json.dumps(payload, indent=2))
    return tmp_json


def _run_inference(
    task_type: str,
    prompt_text: str,
    images: List[str],
    height: int,
    width: int,
    output_num: int,
    num_inference_steps: int,
    guidance_scale: float,
    embedded_cfg_scale: float,
    flow_shift: int,
    flow_reverse: bool,
    seed: int | float | None,
) -> Tuple[str, List[str]]:
    try:
        seed_val = int(seed) if seed is not None else None
    except Exception:
        seed_val = None

    # Hardcode num_frames to 1 for image generation
    num_frames = 1

    ok, message = _check_required_files()
    if not ok:
        return message, []

    if not images:
        return "Please upload at least one reference image.", []
    if len(images) > 4:
        return "Maximum of 4 reference images supported.", []

    tmp_dir = Path(tempfile.mkdtemp())
    saved_images: List[Path] = []
    for uploaded in images:
        src = Path(uploaded)
        dest = tmp_dir / src.name
        shutil.copy(src, dest)
        saved_images.append(dest)

    prompt_json = _write_prompt_json(task_type, prompt_text, saved_images, height, width, output_num)

    output_dir = REPO_ROOT / "outputs" / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=1",
        "--master_addr=127.0.0.1",
        "--master_port=1112",
        "fastvideo/sample/sample_imontage.py",
        "--prompt",
        str(prompt_json),
        "--num_frames",
        str(num_frames),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_inference_steps",
        str(num_inference_steps),
        "--guidance_scale",
        str(guidance_scale),
        "--embedded_cfg_scale",
        str(embedded_cfg_scale),
        "--flow_shift",
        str(flow_shift),
        "--output_path",
        str(output_dir),
        "--model_path",
        DEFAULT_MODEL_ROOT,
        "--dit-weight",
        DEFAULT_DIT_WEIGHT,
        "--vae-tiling",
        "--use-cpu-offload",
    ]
    if flow_reverse:
        cmd.append("--flow-reverse")
    if seed_val is not None:
        cmd.extend(["--seed", str(seed_val)])

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        log = proc.stdout + "\n" + proc.stderr
    except subprocess.CalledProcessError as exc:
        log = exc.stdout + "\n" + exc.stderr
        return f"Inference failed:\n{log}", []

    produced_images = sorted(output_dir.glob("*.png"))
    if not produced_images:
        return f"Inference finished but no images found. Logs:\n{log}", []

    return log, [str(p) for p in produced_images]


def build_demo():
    with gr.Blocks(title="iMontage Inference") as demo:
        gr.Markdown(
            "## iMontage Gradio UI\n"
            "Run the reference inference pipeline with your images and prompt.\n"
            "Ensure model weights are downloaded to `ckpts/` before starting."
        )

        with gr.Row():
            task_type = gr.Dropdown(
                label="Task type",
                choices=[
                    "image_editing",
                    "cref",
                    "conditioned_cref",
                    "sref",
                    "multiview",
                    "storyboard",
                ],
                value="image_editing",
            )
            output_num = gr.Slider(label="Outputs per prompt", minimum=1, maximum=4, value=1, step=1)

        prompt_text = gr.Textbox(
            label="Prompt",
            placeholder="Describe what to generate or edit.",
            lines=3,
        )

        images = gr.Files(label="Reference images (1-4)", file_types=["image"], type="filepath")

        with gr.Row():
            height = gr.Slider(label="Height", minimum=256, maximum=1280, value=1024, step=16)
            width = gr.Slider(label="Width", minimum=256, maximum=1280, value=1024, step=16)

        with gr.Row():
            num_inference_steps = gr.Slider(label="Inference steps", minimum=10, maximum=100, value=50, step=1)
            guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=15.0, value=6.0, step=0.1)
            embedded_cfg_scale = gr.Slider(
                label="Embedded CFG scale", minimum=0.0, maximum=15.0, value=1.0, step=0.1
            )

        with gr.Row():
            flow_shift = gr.Slider(label="Flow shift", minimum=0, maximum=15, value=7, step=1)
            flow_reverse = gr.Checkbox(label="Flow reverse", value=True)
            seed = gr.Number(label="Seed (optional)", value=42, precision=0)

        run_button = gr.Button("Run inference", variant="primary")
        log_output = gr.Textbox(label="Logs", lines=8)
        gallery = gr.Gallery(label="Generated images", preview=True)

        run_button.click(
            fn=_run_inference,
            inputs=[
                task_type,
                prompt_text,
                images,
                height,
                width,
                output_num,
                num_inference_steps,
                guidance_scale,
                embedded_cfg_scale,
                flow_shift,
                flow_reverse,
                seed,
            ],
            outputs=[log_output, gallery],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Gradio share link; useful if the RunPod proxy is unreliable. Use --no-share to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_demo()
    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)
