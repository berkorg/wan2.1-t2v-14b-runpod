#!/usr/bin/env python
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import os
import sys
import logging
import torch
import torch.distributed as dist
from PIL import Image
from datetime import datetime
import warnings
import requests
import uuid
from urllib.parse import urlparse

warnings.filterwarnings("ignore")

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video


def init_distributed(world_size, rank, local_rank):
    """Initialize distributed environment"""
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
        return True
    return False


def init_logging(rank):
    """Initialize logging based on rank"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def init_model_parallel(world_size, ulysses_size, ring_size):
    """Initialize model parallel settings"""
    from xfuser.core.distributed import (
        initialize_model_parallel,
        init_distributed_environment,
    )

    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=ring_size,
        ulysses_degree=ulysses_size,
    )


def generate_video(
    task="i2v-14B",
    size="1280*720",
    ckpt_dir="./Wan2.1-I2V-14B-720P",
    image_path="examples/i2v_input.JPG",
    prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
    frame_num=81,
    sample_steps=40,
    sample_shift=5.0,
    sample_solver="unipc",
    sample_guide_scale=5.0,
    base_seed=-1,
    save_file=None,
    t5_fsdp=True,
    dit_fsdp=True,
    ulysses_size=8,
    ring_size=1,
    t5_cpu=False,
    offload_model=None,
    use_prompt_extend=False,
    prompt_extend_method="local_qwen",
    prompt_extend_model=None,
    prompt_extend_target_lang="ch",
):
    """
    Generate a video from an image and text prompt using Wan model.

    Args:
        task (str): The task to run, e.g., "i2v-14B"
        size (str): The size of the generated video, e.g., "1280*720"
        ckpt_dir (str): Path to the checkpoint directory
        image_path (str): Path to the input image
        prompt (str): Text prompt for content generation
        frame_num (int): Number of frames to generate
        sample_steps (int): Number of diffusion sampling steps
        sample_shift (float): Sampling shift factor for flow matching schedulers
        sample_solver (str): The solver used to sample ('unipc' or 'dpm++')
        sample_guide_scale (float): Classifier-free guidance scale
        base_seed (int): Random seed for generation (-1 for random)
        save_file (str): Path to save the generated video (None for auto-naming)
        t5_fsdp (bool): Whether to use FSDP for T5
        dit_fsdp (bool): Whether to use FSDP for DiT
        ulysses_size (int): Size of the ulysses parallelism in DiT
        ring_size (int): Size of the ring attention parallelism in DiT
        t5_cpu (bool): Whether to place T5 model on CPU
        offload_model (bool): Whether to offload models to CPU during generation
        use_prompt_extend (bool): Whether to use prompt extension
        prompt_extend_method (str): Method for prompt extension
        prompt_extend_model (str): Model for prompt extension
        prompt_extend_target_lang (str): Target language for prompt extension

    Returns:
        str: Path to the saved video file
    """
    # Get distributed environment variables
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    # Initialize logging
    init_logging(rank)

    # Validate task and size
    assert task in WAN_CONFIGS, f"Unsupported task: {task}"
    assert (
        size in SUPPORTED_SIZES[task]
    ), f"Unsupported size {size} for task {task}, supported sizes are: {', '.join(SUPPORTED_SIZES[task])}"

    # Set default offload_model if not specified
    if offload_model is None:
        offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {offload_model}.")

    # Initialize distributed environment
    is_distributed = init_distributed(world_size, rank, local_rank)

    # Validate distributed settings
    if not is_distributed:
        assert not (
            t5_fsdp or dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            ulysses_size > 1 or ring_size > 1
        ), "context parallel are not supported in non-distributed environments."

    # Initialize model parallel if needed
    if ulysses_size > 1 or ring_size > 1:
        assert (
            ulysses_size * ring_size == world_size
        ), f"The number of ulysses_size and ring_size should be equal to the world size."
        init_model_parallel(world_size, ulysses_size, ring_size)

    # Initialize prompt expander if needed
    if use_prompt_extend:
        from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

        if prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=prompt_extend_model, is_vl="i2v" in task
            )
        elif prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=prompt_extend_model, is_vl="i2v" in task, device=rank
            )
        else:
            raise NotImplementedError(
                f"Unsupported prompt_extend_method: {prompt_extend_method}"
            )

    # Get model config
    cfg = WAN_CONFIGS[task]
    if ulysses_size > 1:
        assert (
            cfg.num_heads % ulysses_size == 0
        ), f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(
        f"Generation job config: task={task}, size={size}, ckpt_dir={ckpt_dir}"
    )
    logging.info(f"Generation model config: {cfg}")

    # Synchronize base_seed across processes
    if is_distributed:
        base_seed = [base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        base_seed = base_seed[0]

    # Load and process input image
    logging.info(f"Input prompt: {prompt}")
    logging.info(f"Input image: {image_path}")

    img = Image.open(image_path).convert("RGB")

    # Extend prompt if needed
    if use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                prompt, tar_lang=prompt_extend_target_lang, image=img, seed=base_seed
            )
            if prompt_output.status == False:
                logging.info(f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]

        if is_distributed:
            dist.broadcast_object_list(input_prompt, src=0)

        prompt = input_prompt[0]
        logging.info(f"Extended prompt: {prompt}")

    # Create WanI2V pipeline
    logging.info("Creating WanI2V pipeline.")
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=t5_fsdp,
        dit_fsdp=dit_fsdp,
        use_usp=(ulysses_size > 1 or ring_size > 1),
        t5_cpu=t5_cpu,
    )

    # Generate video
    logging.info("Generating video ...")
    video = wan_i2v.generate(
        prompt,
        img,
        max_area=MAX_AREA_CONFIGS[size],
        frame_num=frame_num,
        shift=sample_shift,
        sample_solver=sample_solver,
        sampling_steps=sample_steps,
        guide_scale=sample_guide_scale,
        seed=base_seed,
        offload_model=offload_model,
    )

    # Save video
    if rank == 0 and video is not None:
        if save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
            save_file = f"{task}_{size}_{ulysses_size}_{ring_size}_{formatted_prompt}_{formatted_time}.mp4"

        logging.info(f"Saving generated video to {save_file}")
        cache_video(
            tensor=video[None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        return save_file

    logging.info("Finished.")
    return None


def generate_video_from_url(
    image_url,
    task="i2v-14B",
    size="1280*720",
    ckpt_dir="./Wan2.1-I2V-14B-720P",
    prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
    frame_num=81,
    sample_steps=40,
    sample_shift=5.0,
    sample_solver="unipc",
    sample_guide_scale=5.0,
    base_seed=-1,
    save_file=None,
    t5_fsdp=True,
    dit_fsdp=True,
    ulysses_size=8,
    ring_size=1,
    t5_cpu=False,
    offload_model=None,
    use_prompt_extend=False,
    prompt_extend_method="local_qwen",
    prompt_extend_model=None,
    prompt_extend_target_lang="ch",
    temp_dir="./temp_images",
):
    """
    Download an image from a URL and generate a video using the Wan model.

    Args:
        image_url (str): URL of the image to download
        temp_dir (str): Directory to save the downloaded image

        All other parameters are passed directly to generate_video()

    Returns:
        str: Path to the saved video file
    """
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)

    # Get rank for logging
    rank = int(os.getenv("RANK", 0))
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )

    # Extract filename from URL or generate a unique name
    parsed_url = urlparse(image_url)
    filename = os.path.basename(parsed_url.path)
    if not filename or "." not in filename:
        # Generate a unique filename with .jpg extension if URL doesn't have a valid filename
        filename = f"{uuid.uuid4()}.jpg"

    local_image_path = os.path.join(temp_dir, filename)

    # Download the image
    if rank == 0:
        logging.info(f"Downloading image from {image_url}")
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            with open(local_image_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logging.info(f"Image saved to {local_image_path}")
        except Exception as e:
            logging.error(f"Error downloading image: {e}")
            raise

    # Synchronize across processes to ensure the image is downloaded before proceeding
    if int(os.getenv("WORLD_SIZE", 1)) > 1:
        dist.barrier()

    # Generate video using the downloaded image
    return generate_video(
        task=task,
        size=size,
        ckpt_dir=ckpt_dir,
        image_path=local_image_path,
        prompt=prompt,
        frame_num=frame_num,
        sample_steps=sample_steps,
        sample_shift=sample_shift,
        sample_solver=sample_solver,
        sample_guide_scale=sample_guide_scale,
        base_seed=base_seed,
        save_file=save_file,
        t5_fsdp=t5_fsdp,
        dit_fsdp=dit_fsdp,
        ulysses_size=ulysses_size,
        ring_size=ring_size,
        t5_cpu=t5_cpu,
        offload_model=offload_model,
        use_prompt_extend=use_prompt_extend,
        prompt_extend_method=prompt_extend_method,
        prompt_extend_model=prompt_extend_model,
        prompt_extend_target_lang=prompt_extend_target_lang,
    )


if __name__ == "__main__":
    # Example usage with an image URL
    # This should be run with: torchrun --nproc_per_node=8 main.py

    # Example image URL - replace with your actual image URL
    image_url = "https://example.com/cat_on_surfboard.jpg"

    generate_video_from_url(
        image_url=image_url,
        task="i2v-14B",
        size="1280*720",
        ckpt_dir="./Wan2.1-I2V-14B-720P",
        prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        t5_fsdp=True,
        dit_fsdp=True,
        ulysses_size=8,
        ring_size=1,
    )
