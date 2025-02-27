import runpod
import os
import sys
import json
import torch
import logging
import subprocess
import time
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError
import tempfile
import uuid
from urllib.parse import urlparse
import multiprocessing

# Import functions from main.py
from main import generate_video, generate_video_from_url

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)


def get_s3_settings():
    """Get S3 settings from environment variables"""
    return {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1"),
        "aws_bucket_name": os.getenv("AWS_BUCKET_NAME"),
    }


class S3Utils:
    """Utility class for S3 operations"""

    def __init__(self, settings):
        self.settings = settings
        self._client = None

    def get_client(self):
        """Get or create S3 client"""
        if self._client is None:
            self._client = boto3.client(
                "s3",
                aws_access_key_id=self.settings["aws_access_key_id"],
                aws_secret_access_key=self.settings["aws_secret_access_key"],
                region_name=self.settings["aws_region"],
            )
        return self._client

    def upload_file(self, file_path, object_name=None):
        """Upload a file to S3 bucket"""
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.get_client().upload_file(
                file_path, self.settings["aws_bucket_name"], object_name
            )
            return True
        except Exception as e:
            logging.error(f"Error uploading file to S3: {e}")
            return False

    def get_s3_url(self, object_name):
        """Get the direct S3 URL for an object"""
        return f"https://{self.settings['aws_bucket_name']}.s3.{self.settings['aws_region']}.amazonaws.com/{object_name}"


def count_available_gpus():
    """Count the number of available GPUs"""
    try:
        return torch.cuda.device_count()
    except:
        return 0


def run_distributed_process(script_args):
    """Run a process with torchrun for distributed execution"""
    cmd = ["torchrun"] + script_args
    logging.info(f"Running command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    # Read and log output in real-time
    for line in iter(process.stdout.readline, ""):
        logging.info(line.strip())

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        logging.error(f"Process exited with code {return_code}")
        return False
    return True


def create_distributed_script(job_input, job_type, output_path):
    """Create a temporary script for distributed execution"""
    script_path = os.path.join(tempfile.gettempdir(), f"{job_type}_{uuid.uuid4()}.py")

    with open(script_path, "w") as f:
        f.write("#!/usr/bin/env python\n")
        f.write("import os\n")
        f.write("import sys\n")
        f.write("import json\n")
        f.write("from main import generate_video, generate_video_from_url\n\n")

        if job_type == "generate-image-to-video":
            f.write("# Parameters from job input\n")
            f.write(f"image_url = {json.dumps(job_input.get('image_url'))}\n")
            f.write(f"prompt = {json.dumps(job_input.get('prompt'))}\n")
            f.write(f"task = {json.dumps(job_input.get('task', 'i2v-14B'))}\n")
            f.write(f"size = {json.dumps(job_input.get('size', '1280*720'))}\n")
            f.write(
                f"ckpt_dir = {json.dumps(job_input.get('ckpt_dir', './Wan2.1-I2V-14B-720P'))}\n"
            )
            f.write(f"frame_num = {job_input.get('frame_num', 81)}\n")
            f.write(f"sample_steps = {job_input.get('sample_steps', 40)}\n")
            f.write(f"sample_shift = {job_input.get('sample_shift', 5.0)}\n")
            f.write(
                f"sample_solver = {json.dumps(job_input.get('sample_solver', 'unipc'))}\n"
            )
            f.write(
                f"sample_guide_scale = {job_input.get('sample_guide_scale', 5.0)}\n"
            )
            f.write(f"base_seed = {job_input.get('base_seed', -1)}\n")
            f.write(f"output_path = {json.dumps(output_path)}\n\n")

            f.write("# Run the generation function\n")
            f.write("result = generate_video_from_url(\n")
            f.write("    image_url=image_url,\n")
            f.write("    task=task,\n")
            f.write("    size=size,\n")
            f.write("    ckpt_dir=ckpt_dir,\n")
            f.write("    prompt=prompt,\n")
            f.write("    frame_num=frame_num,\n")
            f.write("    sample_steps=sample_steps,\n")
            f.write("    sample_shift=sample_shift,\n")
            f.write("    sample_solver=sample_solver,\n")
            f.write("    sample_guide_scale=sample_guide_scale,\n")
            f.write("    base_seed=base_seed,\n")
            f.write("    save_file=output_path,\n")
            f.write("    t5_fsdp=True,\n")
            f.write("    dit_fsdp=True,\n")
            f.write("    ulysses_size=8,\n")  # Use all 8 GPUs for ulysses parallelism
            f.write("    ring_size=1\n")
            f.write(")\n")

        elif job_type == "generate-text-to-video":
            f.write("# Import necessary modules\n")
            f.write("import wan\n")
            f.write("from wan.configs import WAN_CONFIGS, SIZE_CONFIGS\n")
            f.write("from wan.utils.utils import cache_video\n\n")

            f.write("# Parameters from job input\n")
            f.write(f"prompt = {json.dumps(job_input.get('prompt'))}\n")
            f.write(f"task = {json.dumps(job_input.get('task', 't2v-14B'))}\n")
            f.write(f"size = {json.dumps(job_input.get('size', '1280*720'))}\n")
            f.write(
                f"ckpt_dir = {json.dumps(job_input.get('ckpt_dir', './Wan2.1-T2V-14B'))}\n"
            )
            f.write(f"frame_num = {job_input.get('frame_num', 81)}\n")
            f.write(f"sample_steps = {job_input.get('sample_steps', 50)}\n")
            f.write(f"sample_shift = {job_input.get('sample_shift', 5.0)}\n")
            f.write(
                f"sample_solver = {json.dumps(job_input.get('sample_solver', 'unipc'))}\n"
            )
            f.write(
                f"sample_guide_scale = {job_input.get('sample_guide_scale', 5.0)}\n"
            )
            f.write(f"base_seed = {job_input.get('base_seed', -1)}\n")
            f.write(f"output_path = {json.dumps(output_path)}\n\n")

            f.write("# Get rank for distributed processing\n")
            f.write("rank = int(os.getenv('RANK', 0))\n")
            f.write("world_size = int(os.getenv('WORLD_SIZE', 1))\n")
            f.write("local_rank = int(os.getenv('LOCAL_RANK', 0))\n\n")

            f.write("# Get model config\n")
            f.write("cfg = WAN_CONFIGS[task]\n\n")

            f.write("# Initialize model parallel if needed\n")
            f.write("if world_size > 1:\n")
            f.write(
                "    from xfuser.core.distributed import (initialize_model_parallel, init_distributed_environment)\n"
            )
            f.write(
                "    init_distributed_environment(rank=rank, world_size=world_size)\n"
            )
            f.write(
                "    initialize_model_parallel(sequence_parallel_degree=world_size, ring_degree=1, ulysses_degree=8)\n\n"
            )

            f.write("# Create WanT2V pipeline\n")
            f.write("wan_t2v = wan.WanT2V(\n")
            f.write("    config=cfg,\n")
            f.write("    checkpoint_dir=ckpt_dir,\n")
            f.write("    device_id=local_rank,\n")
            f.write("    rank=rank,\n")
            f.write("    t5_fsdp=True,\n")
            f.write("    dit_fsdp=True,\n")
            f.write("    use_usp=(world_size > 1),\n")
            f.write("    t5_cpu=False,\n")
            f.write(")\n\n")

            f.write("# Generate video\n")
            f.write("video = wan_t2v.generate(\n")
            f.write("    prompt,\n")
            f.write("    size=SIZE_CONFIGS[size],\n")
            f.write("    frame_num=frame_num,\n")
            f.write("    shift=sample_shift,\n")
            f.write("    sample_solver=sample_solver,\n")
            f.write("    sampling_steps=sample_steps,\n")
            f.write("    guide_scale=sample_guide_scale,\n")
            f.write("    seed=base_seed,\n")
            f.write("    offload_model=(world_size == 1)\n")
            f.write(")\n\n")

            f.write("# Save video if rank 0\n")
            f.write("if rank == 0 and video is not None:\n")
            f.write("    cache_video(\n")
            f.write("        tensor=video[None],\n")
            f.write("        save_file=output_path,\n")
            f.write("        fps=cfg.sample_fps,\n")
            f.write("        nrow=1,\n")
            f.write("        normalize=True,\n")
            f.write("        value_range=(-1, 1)\n")
            f.write("    )\n")

    os.chmod(script_path, 0o755)  # Make executable
    return script_path


def handle_image_to_video(job_input):
    """Handle image-to-video generation job using multiple GPUs"""
    # Extract parameters from job input
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt")

    # Validate required parameters
    if not image_url:
        return {"error": "image_url is required for image-to-video generation"}
    if not prompt:
        return {"error": "prompt is required for image-to-video generation"}

    # Check available GPUs
    num_gpus = count_available_gpus()
    if num_gpus < 1:
        return {"error": "No GPUs available for video generation"}

    logging.info(f"Found {num_gpus} GPUs available for processing")

    # Generate a unique filename for the output video
    output_filename = f"i2v_{uuid.uuid4()}.mp4"
    temp_output_path = os.path.join(tempfile.gettempdir(), output_filename)

    try:
        # Create a temporary script for distributed execution
        script_path = create_distributed_script(
            job_input, "generate-image-to-video", temp_output_path
        )

        # Determine number of GPUs to use (up to 8)
        num_gpus_to_use = min(8, num_gpus)

        # Run with torchrun for distributed execution
        torchrun_args = [f"--nproc_per_node={num_gpus_to_use}", script_path]

        success = run_distributed_process(torchrun_args)
        if not success or not os.path.exists(temp_output_path):
            return {"error": "Failed to generate video"}

        # Upload to S3 if credentials are available
        s3_settings = get_s3_settings()
        if all(s3_settings.values()):
            s3_utils = S3Utils(s3_settings)

            # Upload the video file
            s3_object_name = f"outputs/{output_filename}"
            upload_success = s3_utils.upload_file(temp_output_path, s3_object_name)

            if upload_success:
                # Get the direct S3 URL
                s3_url = s3_utils.get_s3_url(s3_object_name)

                return {
                    "status": "success",
                    "video_url": s3_url,
                    "message": f"Video generated using {num_gpus_to_use} GPUs and uploaded successfully",
                    "gpus_used": num_gpus_to_use,
                }

            return {
                "status": "partial_success",
                "local_path": temp_output_path,
                "message": f"Video generated using {num_gpus_to_use} GPUs but failed to upload to S3",
                "gpus_used": num_gpus_to_use,
            }
        else:
            # If S3 credentials are not available, return the local path
            return {
                "status": "success",
                "local_path": temp_output_path,
                "message": f"Video generated successfully using {num_gpus_to_use} GPUs (S3 upload not configured)",
                "gpus_used": num_gpus_to_use,
            }

    except Exception as e:
        logging.error(f"Error in image-to-video generation: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        # Clean up temporary script
        if "script_path" in locals() and os.path.exists(script_path):
            try:
                os.remove(script_path)
            except:
                pass


def handle_text_to_video(job_input):
    """Handle text-to-video generation job using multiple GPUs"""
    # Extract parameters from job input
    prompt = job_input.get("prompt")

    # Validate required parameters
    if not prompt:
        return {"error": "prompt is required for text-to-video generation"}

    # Check available GPUs
    num_gpus = count_available_gpus()
    if num_gpus < 1:
        return {"error": "No GPUs available for video generation"}

    logging.info(f"Found {num_gpus} GPUs available for processing")

    # Generate a unique filename for the output video
    output_filename = f"t2v_{uuid.uuid4()}.mp4"
    temp_output_path = os.path.join(tempfile.gettempdir(), output_filename)

    try:
        # Create a temporary script for distributed execution
        script_path = create_distributed_script(
            job_input, "generate-text-to-video", temp_output_path
        )

        # Determine number of GPUs to use (up to 8)
        num_gpus_to_use = min(8, num_gpus)

        # Run with torchrun for distributed execution
        torchrun_args = [f"--nproc_per_node={num_gpus_to_use}", script_path]

        success = run_distributed_process(torchrun_args)
        if not success or not os.path.exists(temp_output_path):
            return {"error": "Failed to generate video"}

        # Upload to S3 if credentials are available
        s3_settings = get_s3_settings()
        if all(s3_settings.values()):
            s3_utils = S3Utils(s3_settings)

            # Upload the video file
            s3_object_name = f"outputs/{output_filename}"
            upload_success = s3_utils.upload_file(temp_output_path, s3_object_name)

            if upload_success:
                # Get the direct S3 URL
                s3_url = s3_utils.get_s3_url(s3_object_name)

                return {
                    "status": "success",
                    "video_url": s3_url,
                    "message": f"Video generated using {num_gpus_to_use} GPUs and uploaded successfully",
                    "gpus_used": num_gpus_to_use,
                }

            return {
                "status": "partial_success",
                "local_path": temp_output_path,
                "message": f"Video generated using {num_gpus_to_use} GPUs but failed to upload to S3",
                "gpus_used": num_gpus_to_use,
            }
        else:
            # If S3 credentials are not available, return the local path
            return {
                "status": "success",
                "local_path": temp_output_path,
                "message": f"Video generated successfully using {num_gpus_to_use} GPUs (S3 upload not configured)",
                "gpus_used": num_gpus_to_use,
            }

    except Exception as e:
        logging.error(f"Error in text-to-video generation: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        # Clean up temporary script
        if "script_path" in locals() and os.path.exists(script_path):
            try:
                os.remove(script_path)
            except:
                pass


def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]
    job_type = job_input.get("job_type")

    if job_type is None:
        return {"error": "You need to specify job_type"}

    if job_type == "generate-image-to-video":
        return handle_image_to_video(job_input)
    elif job_type == "generate-text-to-video":
        return handle_text_to_video(job_input)
    elif job_type == "test_job":
        return {"status": "handler is fine!"}
    else:
        return {
            "error": "job_type should be one of 'generate-image-to-video', 'generate-text-to-video', or 'test_job'"
        }


# Start the serverless function
runpod.serverless.start({"handler": handler})
