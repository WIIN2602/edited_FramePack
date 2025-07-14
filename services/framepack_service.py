import torch
import einops
import safetensors.torch as sf
import numpy as np
import openai
from dotenv import load_dotenv
import os
from PIL import Image
import requests
import base64
import uuid

from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel,
    CLIPTextModel,
    LlamaTokenizerFast,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel
)

from diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_decode,
    vae_encode,
    vae_decode_fake
)
from diffusers_helper.utils import (
    save_bcthw_as_mp4,
    crop_or_pad_yield_mask,
    soft_append_bcthw,
    resize_and_center_crop,
    state_dict_weighted_merge,
    state_dict_offset_merge,
    generate_timestamp
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu,
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device,
    DynamicSwapInstaller,
    unload_complete_models,
    load_model_as_complete
)
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

from utils import *
from utils.prompt_utils import get_prompt_by_category
from uuid import uuid4
from typing import List, Optional
from dbcontroller.supabase import SupabaseConnector

from warnings import filterwarnings
filterwarnings("ignore")


load_dotenv()  

client = openai.OpenAI(api_key=os.getenv("GPT_TOKEN"))
supabase = SupabaseConnector()

SERVICE_DIR = os.getenv("SERVICE_DIR")
IMAGE_DIR = os.getenv("IMAGE_DIR")

def gen_ImgPrompt(session_id: str):
    try:
        script_data = supabase.find_script(session_id)
        if not script_data:
            msg = f"[ERROR] gen_ImgPrompt: No script found for session ID: {session_id}"
            print(msg)
            raise ValueError(msg)

        task1_path = os.path.join(SERVICE_DIR, SCRIPT_DIR, session_id, f"{session_id}_task1.txt")
        print(f"[DEBUG] gen_ImgPrompt: Task1 path: {task1_path}")

        if not script_data.get('summed'):
            msg = f"[ERROR] gen_ImgPrompt: Script for session ID: {session_id} is not summarized"
            print(msg)
            raise ValueError(msg)

        try:
            with open(task1_path, 'r', encoding='utf-8') as file:
                summary_text = file.read()
        except FileNotFoundError:
            msg = f"[ERROR] gen_ImgPrompt: Summary file not found at path: {task1_path}"
            print(msg)
            raise 
        except Exception as e:
            print(f"[ERROR] gen_ImgPrompt: Failed to read summary file: {e}")
            raise

        print(f"[DEBUG] gen_ImgPrompt: Summary for session {session_id}:\n{summary_text}")
        print("[DEBUG] gen_ImgPrompt: Calling OpenAI API to generate image prompt...")

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                temperature=0.7,
                max_tokens=400,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a world-class prompt engineer for text-to-image AI models "
                            "(e.g. Stable Diffusion, Midjourney, DALL·E). "
                            "Your job is to turn a plain summary into a vivid, detailed, and "
                            "unambiguous prompt that can be copied & pasted straight into an "
                            "image generation pipeline."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Below is a short summary of the scene or concept I want illustrated. "
                            "Please generate a single, ready-to-use text-to-image prompt that includes:\n"
                            "  • Main subject(s) and environment\n"
                            "  • Composition (angles, framing)\n"
                            "  • Mood and atmosphere\n"
                            "  • Color palette and lighting\n"
                            "  • Artistic style (e.g. hyper-realistic, watercolor, cinematic)\n"
                            "  • Camera settings (e.g. focal length, depth of field)\n"
                            "  • Any relevant textures or details\n"
                            "  • A concise negative-prompt list (if supported)\n\n"
                            f"Summary:\n{summary_text}"
                        )
                    }
                ]
            )
            img_prompt = response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] gen_ImgPrompt: Failed to generate prompt using OpenAI API: {e}")
            raise

        print(f"[DEBUG] gen_ImgPrompt: Generated image prompt: {img_prompt}")

        try:
            check_exit = supabase.find_img_prompt(session_id)
        except Exception as e:
            print(f"[ERROR] gen_ImgPrompt: Failed to check for existing image prompt in database: {e}")
            raise

        prompt_infos = []
        random_uid = str(uuid4())
        try:
            if not check_exit:
                print("[DEBUG] gen_ImgPrompt: No existing prompt found in database. Inserting new prompt...")
                prompt_infos.append({
                    "uid": random_uid,
                    "session_id": session_id,
                    "image_prompt": img_prompt,
                    "framepack_prompt": None,
                    "image_name": None,
                    "video_name": None
                })
                supabase.insert_prompt(prompt_infos)
                print("[DEBUG] gen_ImgPrompt: Prompt inserted into database successfully.")
            else:
                print("[DEBUG] gen_ImgPrompt: Existing prompt found in database. Updating prompt...")
                prompt_infos.append({
                    "image_prompt": img_prompt
                })
                supabase.update_framepack_prompt(session_id, prompt_infos)
                print("[DEBUG] gen_ImgPrompt: Prompt updated into database successfully.")
        except Exception as e:
            print(f"[ERROR] gen_ImgPrompt: Failed to insert or update prompt in database: {e}")
            raise

        return img_prompt

    except ValueError as ve:
        print(f"[EXCEPTION] gen_ImgPrompt: ValueError: {ve}")
    except Exception as e:
        print(f"[EXCEPTION] gen_ImgPrompt: Unexpected error: {e}")

def gen_Image(session_id: str) -> str:
    img_path = None
    try:
        # Initialize OpenAI client
        if client:
            print(f"[DEBUG] gen_image: OpenAI client activate")
        else :
            print(f"[DEBUG] gen_image: OpenAI client not activate")

        # Retrieve prompt from DB
        check_prompt = supabase.find_img_prompt(session_id)
        if not check_prompt:
            print(f"[ERROR] gen_image: No prompt found for session ID: {session_id}")
        prompt = check_prompt['image_prompt']
        show_prompt = prompt[:100]
        print(f"[DEBUG] gen_image: image prompt is {show_prompt}...")

        # Generate image with OpenAI
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        show_url = image_url[:100]
        print(f"[DEBUG] gen_image: image url is {show_url}...")

        # Download image
        img_data = requests.get(image_url).content
        if not img_data:
            raise ValueError("Downloaded image data is empty.")
        print(f"[DEBUG] gen_image: Image downloaded successfully")

        # Save image to disk
        img_folder = os.path.join(SERVICE_DIR, IMAGE_DIR, session_id)
        os.makedirs(img_folder, exist_ok=True)
        img_name = f"{session_id[:8]}.jpg"
        img_path = os.path.join(img_folder, img_name)

        with open(img_path, "wb") as f:
            f.write(img_data)
        print(f"[DEBUG] Image saved at {img_path}")

        try:
            print("[DEBUG] gen_image: Updating prompt in database...")
            prompt_infos = []
            prompt_infos.append({
                "image_name": img_name
            })
            supabase.update_framepack_prompt(session_id, prompt_infos)
            print("[DEBUG] gen_image: Prompt updated successfully.")
        except Exception as e:
            print(f"[ERROR] gen_image: Failed to insert or update prompt in database: {e}")
            raise

    except ValueError as ve:
        print(f"[EXCEPTION] gen_image: {ve}")
    except Exception as e:
        print(f"[EXCEPTION] gen_image: Unexpected error: {e}")

    return img_path

def gen_FramePackPrompt(session_id: str):
    try:
        script_data = supabase.find_script(session_id)
        if not script_data:
            msg = f"[ERROR] gen_FramePackPrompt: No script found for session ID: {session_id}"
            print(msg)
            raise ValueError(msg)

        task1_path = os.path.join(SERVICE_DIR, SCRIPT_DIR, session_id, f"{session_id}_task1.txt")
        print(f"[DEBUG] gen_FramePackPrompt: Task1 path: {task1_path}")

        if not script_data.get('summed'):
            msg = f"[ERROR] gen_FramePackPrompt: Script for session ID: {session_id} is not summarized"
            print(msg)
            raise ValueError(msg)

        try:
            img_folder = os.path.join(SERVICE_DIR, IMAGE_DIR, session_id)
            os.makedirs(img_folder, exist_ok=True)
            img_name = f"{session_id[:8]}.jpg"
            img_path = os.path.join(img_folder, img_name)
            with open(img_path, "rb") as image_file:  # Chang this line before run
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            task1_path = os.path.join(SERVICE_DIR, SCRIPT_DIR, session_id, f"{session_id}_task1.txt")
            print(f"[DEBUG] gen_ImgPrompt: Task1 path: {task1_path}")

            with open(task1_path, 'r', encoding='utf-8') as file:
                summary_text = file.read()
        except FileNotFoundError:
            msg = f"[ERROR] gen_FramePackPrompt: Summary file not found at path: {task1_path}"
            print(msg)
            raise 
        except Exception as e:
            print(f"[ERROR] gen_FramePackPrompt: Failed to read summary file: {e}")
            raise

        print(f"[DEBUG] gen_FramePackPrompt: Summary for session {session_id}:\n{summary_text}")
        print("[DEBUG] gen_FramePackPrompt: Calling OpenAI API to generate framepack prompt...")

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                max_tokens=400,
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a world-class prompt engineer for FramPack-compatible AI models. "
                        "These models focus on generating structured images suitable for video generation, "
                        "based on consistent spatial layouts and semantic object relationships.\n\n"
                        "Given a reference image and a short summary of scene evolution, your job is to generate "
                        "a clear, structured prompt that describes how the scene should change over time — "
                        "suitable for generating a video sequence from static images."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Here is the summary of the dynamic elements that should evolve across the video:\n"
                                f"{summary_text}\n\n"
                                "Please generate a structured prompt that includes:\n"
                                "  • Key objects in the reference image that should stay consistent\n"
                                "  • Objects or environmental elements that will change or move\n"
                                "  • Description of motion, transformation, or timeline-based evolution\n"
                                "  • Spatial relationships between elements (before/after movement)\n"
                                "  • Environmental factors (day to night, weather, lighting)\n"
                                "  • No camera or artistic style settings — only layout, movement, and function\n"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            )
            framepack_prompt = response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] gen_FramePackPrompt: Failed to generate prompt using OpenAI API: {e}")
            raise

        print(f"[DEBUG] gen_FramePackPrompt: Generated framepack prompt: {framepack_prompt}")

        prompt_infos = []
        random_uid = str(uuid4())
        try:
            print("[DEBUG] gen_FramePackPrompt: Updating prompt in database...")
            prompt_infos.append({
                "framepack_prompt": framepack_prompt
            })
            supabase.update_framepack_prompt(session_id, prompt_infos)
            print("[DEBUG] gen_FramePackPrompt: Prompt updated successfully.")
        except Exception as e:
            print(f"[ERROR] gen_FramePackPrompt: Failed to insert or update prompt in database: {e}")
            raise

        return framepack_prompt

    except ValueError as ve:
        print(f"[EXCEPTION] gen_FramePackPrompt: ValueError: {ve}")
    except Exception as e:
        print(f"[EXCEPTION] gen_FramePackPrompt: Unexpected error: {e}")

@torch.no_grad()
def run_Framepack(session_id: str, total_second_length: int, latent_window_size: int):
    result = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

        # Retrieve prompt data
        try:
            prompt_data = supabase.find_img_prompt(session_id)
            if not prompt_data:
                raise ValueError(f"No prompt found for session ID: {session_id}")
            framepack_prompt = prompt_data.get("framepack_prompt")
            image_name = prompt_data.get("image_name")
            if not framepack_prompt or not image_name:
                raise ValueError(f"Missing framepack prompt or image for session ID: {session_id}")
            print(f"[DEBUG] run_framepack: Retrieved prompt and image name.")
        except Exception as e:
            print(f"[ERROR] run_framepack: Failed to retrieve prompt data - {e}")
            raise

        # Load image
        try:
            img_path = os.path.join(SERVICE_DIR, IMAGE_DIR, session_id, image_name)
            with open(img_path, "rb") as f:
                read_input_image = Image.open(f).convert("RGB")
            print(f"[DEBUG] run_framepack: Loaded image from {img_path}")
        except FileNotFoundError:
            print(f"[ERROR] run_framepack: Image file not found at path {img_path}")
            raise
        except Exception as e:
            print(f"[ERROR] run_framepack: Failed to load image - {e}")
            raise

        # Define parameters
        try:
            seed = 313447
            steps = 10
            cfg = 1
            gs = 10
            rs = 0.0
            gpu_memory_preservation = 10
            use_teacache_input = "y"
            use_teacache = "y"
            mp4_crf = 16
            n_prompt = ""
            prompt = framepack_prompt
            input_image_path = img_path
            total_second_length = total_second_length
            latent_window_size = latent_window_size
            print(f"[DEBUG] run_framepack: Parameters prepared.")
        except Exception as e:
            print(f"[ERROR] run_framepack: Failed to prepare parameters - {e}")
            raise

        # Setup Environment
        try:
            gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[DEBUG] run_framepack: Using device: {gpu}")
        except Exception as e:
            print(f"[ERROR] run_framepack: Device configuration failed - {e}")
            raise

        # Run Framepack Model
        random_uid = str(uuid4())
        try:
            try: 
                output_dir = os.path.join(SERVICE_DIR, "framepack_outputs", session_id)
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to create output path :\n {e}")
                raise

            # Load models
            try:
                text_encoder = LlamaModel.from_pretrained(
                    "hunyuanvideo-community/HunyuanVideo",
                    subfolder='text_encoder',
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).cpu().eval()
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to load text_encoder - {e}")
                raise
            try:
                text_encoder_2 = CLIPTextModel.from_pretrained(
                    "hunyuanvideo-community/HunyuanVideo",
                    subfolder='text_encoder_2',
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).cpu().eval()
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to load text_encoder_2 - {e}")
                raise
            try:
                tokenizer = LlamaTokenizerFast.from_pretrained(
                    "hunyuanvideo-community/HunyuanVideo",
                    subfolder='tokenizer'
                )
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to load LlamaTokenizerFast - {e}")
                raise
            try:
                tokenizer_2 = CLIPTokenizer.from_pretrained(
                    "hunyuanvideo-community/HunyuanVideo",
                    subfolder='tokenizer_2'
                )
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to load CLIPTokenizer - {e}")
                raise
            try:
                vae = AutoencoderKLHunyuanVideo.from_pretrained(
                    "hunyuanvideo-community/HunyuanVideo",
                    subfolder='vae',
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).cpu().eval()
                vae.enable_slicing()
                vae.enable_tiling()
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to load VAE model - {e}")
                raise
            try:
                feature_extractor = SiglipImageProcessor.from_pretrained(
                    "lllyasviel/flux_redux_bfl",
                    subfolder='feature_extractor'
                )
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to load feature_extractor - {e}")
                raise
            try:
                image_encoder = SiglipVisionModel.from_pretrained(
                    "lllyasviel/flux_redux_bfl",
                    subfolder='image_encoder',
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).cpu().eval()
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to load image_encoder - {e}")
                raise
            try:
                transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                    'lllyasviel/FramePackI2V_HY',
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                ).cpu().eval()
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to load transformer model - {e}")
                raise
            try:
                DynamicSwapInstaller.install_model(transformer, device=device)
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to install transformer to device - {e}")
                raise
            try:
                DynamicSwapInstaller.install_model(text_encoder, device=device)
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to install text_encoder to device - {e}")
                raise
            try:
                transformer.high_quality_fp32_output_for_inference = True
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to set FP32 output mode - {e}")
                raise
            try:
                unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed to unload models - {e}")
                raise

            # text encoding
            try:
                fake_diffusers_current_device(text_encoder, gpu)
                load_model_as_complete(text_encoder_2, target_device=gpu)

                llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

                if cfg == 1:
                    llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
                else:
                    llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
                llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed in text encoding - {e}")
                raise

            # input image encoding
            try:
                image_id = uuid.uuid4()
                input_image = read_input_image
                W, H = input_image.size
                height, width = find_nearest_bucket(H, W, resolution=640)
                input_image_np = resize_and_center_crop(input_image_path, target_width=width, target_height=height)

                Image.fromarray(input_image_np).save(os.path.join(output_dir, f'{image_id[:8]}.png'))

                input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
                input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed in input image encoding - {e}")
                raise

            # VAE encoding
            try:
                load_model_as_complete(vae, target_device=gpu)
                start_latent = vae_encode(input_image_pt, vae)

                # CLIP Vision
                load_model_as_complete(image_encoder, target_device=gpu)
                image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
                image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed in VAE encoding - {e}")
                raise

            # Dtype
            try:
                llama_vec = llama_vec.to(transformer.dtype)
                llama_vec_n = llama_vec_n.to(transformer.dtype)
                clip_l_pooler = clip_l_pooler.to(transformer.dtype)
                clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
                image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
            except Exception as e:
                print(f"[ERROR] run_framepack: Failed in Dtype - {e}")
                raise

            # Sampling
            total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
            total_latent_sections = int(max(round(total_latent_sections), 1))

            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3

            history_latents = torch.zeros(
                size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
                dtype=torch.float32
            ).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            latent_paddings = reversed(range(total_latent_sections))

            if total_latent_sections > 4:
                # Adjust padding pattern when section count is high
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            for latent_padding in latent_paddings:
                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size

                print(f'[DEBUG] latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                (
                    clean_latent_indices_pre,
                    blank_indices,
                    latent_indices,
                    clean_latent_indices_post,
                    clean_latent_2x_indices,
                    clean_latent_4x_indices
                ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)

                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                unload_complete_models()
                move_model_to_device_with_memory_preservation(
                    transformer,
                    target_device=gpu,
                    preserved_memory_gb=gpu_memory_preservation
                )

                transformer.initialize_teacache(
                    enable_teacache=use_teacache,
                    num_steps=steps
                )

                def callback(d):
                    preview = vae_decode_fake(d['denoised'])
                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling {current_step}/{steps}'
                    desc = (
                        f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, '
                        f'Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). '
                        'The video is being extended now ...'
                    )

                unload_complete_models()
                torch.cuda.empty_cache()

                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

                if is_last_section:
                    generated_latents = torch.cat(
                        [start_latent.to(generated_latents), generated_latents],
                        dim=2
                    )

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                offload_model_from_device_for_memory_preservation(
                    transformer,
                    target_device=gpu,
                    preserved_memory_gb=8
                )
                load_model_as_complete(vae, target_device=gpu)

                real_history_latents = history_latents[:, :, :total_generated_latent_frames]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3
                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                unload_complete_models()

                print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
                if is_last_section:
                    output_filename = os.path.join(output_dir, f'{image_id}_{total_generated_latent_frames}.mp4')
                    save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
                    break

            Image.save(os.path.join(output_dir, "reference_image.jpg"))
            result = output_filename
            print(f"[DEBUG] run_framepack: Framepack data saved at {result}")
        except Exception as e:
            print(f"[ERROR] run_framepack: Framepack failed - {e}")
            raise
        
        # Clear momory
        import gc

        def clear_gpu_memory():
            torch.cuda.empty_cache()
            gc.collect()
            print("[DEBUG] GPU memory cleared.")
        clear_gpu_memory()

        prompt_infos = []
        # Update database
        try:
            print("[DEBUG] run_Framepack: Updating prompt in database...")
            prompt_infos.append({
                "video_name": "Test run framepack service"
            })
            supabase.update_framepack_prompt(session_id, prompt_infos)
            print("[DEBUG] run_Framepack: Prompt updated successfully.")
        except Exception as e:
            print(f"[ERROR] run_framepack: Failed to update database - {e}")
            raise

    except ValueError as ve:
        print(f"[EXCEPTION] run_framepack: ValueError: {ve}")
    except Exception as e:
        print(f"[EXCEPTION] run_framepack: Unexpected error: {e}")

    return result
