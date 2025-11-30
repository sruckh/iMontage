import argparse
import os
import copy
from pathlib import Path
from collections.abc import Iterable
from torchvision.transforms import transforms, functional
import json

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange

from fastvideo.models.hunyuan.inference import HunyuanVideoSampler
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
from fastvideo.utils.prompt_template import get_prompt

def initialize_distributed():
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)


def resize_to_desired_aspect_ratio(video, aspect_ratio, aspect_size, random_shift=False):
    ## video is in shape [f, h, w, c]
    height, width = video.shape[1:3]
        
    # # resize
    aspect_ratio_fact = width / height
    if isinstance(aspect_ratio, Iterable):
        bucket_idx = np.argmin(np.abs(aspect_ratio_fact - aspect_ratio))
        aspect_ratio = aspect_ratio[bucket_idx]
        target_size_height, target_size_width = aspect_size[bucket_idx]
    else:
        bucket_idx = 0
        target_size_height, target_size_width = aspect_size
    
    if aspect_ratio_fact < aspect_ratio:
        scale = target_size_width / width
    else:
        scale = target_size_height / height

    width_scale = int(round(width * scale))
    height_scale = int(round(height * scale))


    # # crop
    delta_h = height_scale - target_size_height
    delta_w = width_scale - target_size_width
    assert delta_w>=0
    assert delta_h>=0
    assert not all(
        [delta_h, delta_w]
    )  

    if random_shift:
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    else:
        top = delta_h//2
        left = delta_w//2

    ## resize image and crop
    resize_crop_transform = transforms.Compose([
        transforms.Resize((height_scale, width_scale)),
        lambda x: functional.crop(x, top, left, target_size_height, target_size_width),
    ])

    video = rearrange(video, 't h w c -> t c h w')
    video = torch.stack([resize_crop_transform(frame) for frame in video], dim=0)

    return video, bucket_idx
################################################################################

def fetch_json_prompt(args):
    prompts_list = []
    with open(args.prompt) as f:
        prompts = json.load(f)
    for _, item in prompts.items():
        try:
            task_type = item["task_type"]
        except KeyError:
            raise KeyError("Missing required field: task_type") from None
        instr = item["prompts"]
        imgs_path = item["images"]
        height = item.get("height", args.height)
        width = item.get("width", args.width)
        try:
            output_num = item["output_num"]
        except KeyError:
            raise KeyError("Missing required field: output_num") from None
        
        instr = get_prompt(output_num, task_type, instr)
        print(f"After prompt reformating: {instr}")
        
        prompts_list.append([instr, imgs_path, (height, width)])
    return prompts_list, task_type
        
def main(args):
    models_root_path = Path(args.model_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

    args = hunyuan_video_sampler.args

    # NOTE We only support json prompt file for now
    assert isinstance(args.prompt, str) and args.prompt.endswith('.json'), \
        f"Expected args.prompt to be a JSON file, got {args.prompt}"

    prompts_list, task = fetch_json_prompt(args)
        
    prompts = prompts_list
        
    if nccl_info.rank_within_group == 0:
        os.makedirs(args.output_path, exist_ok=True)

    for idx, item in enumerate(prompts):
        prompt = item[0]
        images = item[1]
        height, width = item[2]
        image_save_paths = []
        
        import re
        match = re.search(r"Please output (\d+) images", prompt)
        if match:
            num_output = int(match.group(1))
        else:
            # Fallback or default if pattern not found
            print(f"Warning: Could not parse output number from prompt '{prompt}'. Defaulting to 1.")
            num_output = 1
            
        print(f"output num: {num_output}")
        images = images.split(',') if isinstance(images, str) else images
        images_name = [os.path.basename(image).split('.')[0] for image in images]
        # Use modulo to cycle through names if we have fewer names than outputs, or just use the first one base name with index
        base_name = images_name[0]
        for idx in range(num_output):
            image_save_paths += [os.path.join(args.output_path, f"{base_name}_{idx}.png")]

        img_input = []
        
        for idx, image_path in enumerate(images):
            image = imageio.imread(image_path)

            if image.shape[-1] == 4:  # RGBA
                alpha = image[..., 3:4].astype('float32') / 255.0  # [H, W, 1]
                rgb = image[..., :3].astype('float32')
                white = 255.0
                comp = rgb * alpha + white * (1.0 - alpha)
                image = comp.astype('uint8')
            else:
                if image.ndim == 2: # grayscale
                    image = np.stack([image] * 3, axis=-1)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # [C, H, W]

            image = image.unsqueeze(0)
            if idx == 0:
                _, C, H, W = image.shape
            img_input.append(image)

        # Ensure height and width are multiples of 16 by center-cropping to the nearest lower multiple
        # NOTE: we crop (not pad). If the dimension is smaller than 16 we skip cropping.
        def _center_crop_to_multiple(tensor, multiple=16):
            # tensor: [B, C, H, W]
            _, _, h, w = tensor.shape
            target_h = max((h // multiple) * multiple, multiple) if h >= multiple else h
            target_w = max((w // multiple) * multiple, multiple) if w >= multiple else w

            if target_h == h and target_w == w:
                return tensor, h, w, (0, 0)

            if target_h > h or target_w > w:
                return tensor, h, w, (0, 0)

            top = (h - target_h) // 2
            left = (w - target_w) // 2
            cropped = tensor[:, :, top: top + target_h, left: left + target_w]
            return cropped, target_h, target_w, (top, left)

        image, H_new, W_new, _ = _center_crop_to_multiple(image, multiple=16)
        # update H, W only if they changed
        if H_new != H or W_new != W:
            H, W = H_new, W_new
            # overwrite the original image file with the cropped version so downstream code sees the same size
            try:
                # image is [1, C, H, W] with floats in [0,1]
                img_to_save = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype('uint8')
                # Use the original path variable `image_path` to overwrite
                imageio.imsave(image_path, img_to_save)
                if nccl_info.rank_within_group == 0:
                    print(f"Center-cropped and saved image to {image_path} with size {W}x{H}")
            except Exception as e:
                if nccl_info.rank_within_group == 0:
                    print(f"Warning: failed to overwrite {image_path} after cropping: {e}")

        outputs = hunyuan_video_sampler.predict(
            prompt=prompt,
            video=img_input,
            height=height,
            width=width,
            video_length=args.num_frames,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            media_type="image",
            output_num=len(image_save_paths),
        )
        print(f"predict done,  rank: {nccl_info.rank_within_group}")
        if nccl_info.rank_within_group ==0: 
            for i in range(len(image_save_paths)):
                image_save_path = image_save_paths[i]
                image = rearrange(outputs["samples"][i].squeeze(0), " c h w -> c h w")
                image = (image * 255).to(torch.uint8).cpu()
                image = rearrange(image, "c h w -> h w c").numpy()
                imageio.imsave(image_save_path, image)
                print(f"Image saved to {image_save_path}")
            
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--prompt", type=str, help="prompt file for inference")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="data/hunyuan")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--file_name", type=str, default="video")
    parser.add_argument("--fps", type=int, default=24)

    # Additional parameters
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument("--neg_prompt", type=str, default=None, help="Negative prompt for sampling.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=6.0,
        help="Embedded classifier free guidance scale.",
    )
    parser.add_argument("--flow_shift", type=int, default=7, help="Flow shift parameter.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate per prompt.",
    )
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    parser.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )
    parser.add_argument(
        "--dit-weight",
        type=str,
        default="data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument("--flow-solver", type=str, default="euler", help="Solver for flow matching.")
    parser.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help=
        "Use linear quadratic schedule for flow matching. Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    parser.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--rope-theta", type=int, default=256, help="Theta used in RoPE.")

    parser.add_argument("--vae", type=str, default="884-16c-hy")
    parser.add_argument("--vae-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vae-tiling", action="store_true", default=True)
    parser.add_argument("--vae-sp", action="store_true", default=False)

    parser.add_argument("--text-encoder", type=str, default="llm")
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="llm")
    parser.add_argument("--prompt-template", type=str, default="dit-llm-encode")
    parser.add_argument("--prompt-template-video", type=str, default="dit-llm-encode-video")
    parser.add_argument("--hidden-state-skip-layer", type=int, default=2)
    parser.add_argument("--apply-final-norm", action="store_true")

    parser.add_argument("--text-encoder-2", type=str, default="clipL")
    parser.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--enable_torch_compile",
        action="store_true",
        help="Use torch.compile for speeding up STA inference without teacache",
    )
    parser.add_argument("--text-states-dim-2", type=int, default=768)
    parser.add_argument("--tokenizer-2", type=str, default="clipL")
    parser.add_argument("--text-len-2", type=int, default=77)
    
    args = parser.parse_args()
    # process for vae sequence parallel
    if args.vae_sp and not args.vae_tiling:
        raise ValueError("Currently enabling vae_sp requires enabling vae_tiling, please set --vae-tiling to True.")
    
    initialize_distributed()

    main(args)