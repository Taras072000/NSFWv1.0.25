import os
import time
import base64
import io
from PIL import Image, ImageDraw
import boto3
from botocore.config import Config
import runpod
import logging
import math
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler

def get_env(name, default=None):
    v = os.getenv(name)
    return v if v else default

level = os.getenv("LOG_LEVEL", "info").lower()
lvl = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}.get(level, logging.INFO)
logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(message)s", force=True)
dsn = os.getenv("SENTRY_DSN")
if dsn:
    sentry_sdk.init(dsn=dsn, traces_sample_rate=0.0, integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)])
logger = logging.getLogger("worker")
PIPE_BASE = None
PIPE_REFINER = None

def load_pipelines():
    global PIPE_BASE, PIPE_REFINER
    if PIPE_BASE is None:
        model_id = get_env("GEN_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
        token = get_env("HF_TOKEN")
        PIPE_BASE = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            token=token,
            add_watermarker=False,
        ).to("cuda")
        try:
            PIPE_BASE.scheduler = DPMSolverMultistepScheduler.from_config(PIPE_BASE.scheduler.config)
        except Exception:
            pass
        try:
            PIPE_BASE.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        try:
            PIPE_BASE.enable_vae_slicing()
            PIPE_BASE.enable_attention_slicing()
        except Exception:
            pass
    use_refiner = get_env("GEN_ENABLE_REFINER", "false").lower() == "true"
    if use_refiner and PIPE_REFINER is None:
        ref_id = get_env("GEN_REFINER_MODEL_ID", "stabilityai/stable-diffusion-xl-refiner-1.0")
        token = get_env("HF_TOKEN")
        PIPE_REFINER = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            ref_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            token=token,
        ).to("cuda")
        try:
            PIPE_REFINER.scheduler = DPMSolverMultistepScheduler.from_config(PIPE_REFINER.scheduler.config)
        except Exception:
            pass
        try:
            PIPE_REFINER.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        try:
            PIPE_REFINER.enable_vae_slicing()
            PIPE_REFINER.enable_attention_slicing()
        except Exception:
            pass

def upload_s3(img_bytes, file_name):
    if (get_env("SAVE_OUTPUTS_TO_S3", "false").lower() != "true"):
        return None
    endpoint = get_env("S3_ENDPOINT")
    region = get_env("S3_REGION")
    bucket = get_env("S3_BUCKET")
    access = get_env("S3_ACCESS_KEY_ID")
    secret = get_env("S3_SECRET_ACCESS_KEY")
    base = get_env("S3_PUBLIC_BASE_URL")
    if not bucket or not access or not secret:
        return None
    prefix = get_env("S3_KEY_PREFIX", "images").strip("/")
    key = f"{prefix}/{file_name}" if prefix else file_name
    session = boto3.session.Session(aws_access_key_id=access, aws_secret_access_key=secret, region_name=region)
    force_path = get_env("S3_FORCE_PATH_STYLE", "false").lower() == "true"
    cfg = Config(s3={"addressing_style": "path" if force_path else "auto"})
    client = session.client("s3", endpoint_url=endpoint, config=cfg) if endpoint else session.client("s3", config=cfg)
    kwargs = {"Bucket": bucket, "Key": key, "Body": img_bytes, "ContentType": "image/png"}
    cache_control = get_env("S3_CACHE_CONTROL")
    if cache_control:
        kwargs["CacheControl"] = cache_control
    require_acl = get_env("S3_REQUIRE_ACL", "false").lower() == "true"
    if require_acl:
        kwargs["ACL"] = "public-read"
    client.put_object(**kwargs)
    if base:
        return f"{base.rstrip('/')}/{key}"
    if region:
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def generate_sdxl(prompt, negative_prompt, width, height, steps, guidance, seed, use_refiner, refiner_steps):
    load_pipelines()
    w = clamp(int(width), 64, 4096)
    h = clamp(int(height), 64, 4096)
    max_px = int(get_env("MAX_IMAGE_PIXELS", str(1920*1080)))
    if w * h > max_px:
        scale = math.sqrt(max_px / float(w * h))
        w = max(64, int(w * scale))
        h = max(64, int(h * scale))
    gen = torch.Generator(device="cuda")
    if seed:
        try:
            gen = gen.manual_seed(int(seed))
        except:
            pass
    if use_refiner and PIPE_REFINER is not None:
        latents = PIPE_BASE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=w,
            height=h,
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            output_type="latent",
            generator=gen,
            denoising_end=0.8,
        ).images
        image = PIPE_REFINER(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=latents,
            guidance_scale=float(guidance),
            num_inference_steps=int(refiner_steps),
            generator=gen,
            denoising_start=0.8,
        ).images[0]
    else:
        image = PIPE_BASE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=w,
            height=h,
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=gen,
        ).images[0]
    b = io.BytesIO()
    image.save(b, format="PNG")
    return b.getvalue(), w, h

def handler(event):
    inp = event.get("input") if isinstance(event, dict) else event
    prompt = (inp or {}).get("prompt")
    width = (inp or {}).get("width") or int(get_env("GEN_IMAGE_WIDTH", "1920"))
    height = (inp or {}).get("height") or int(get_env("GEN_IMAGE_HEIGHT", "1080"))
    steps = int((inp or {}).get("steps") or int(get_env("GEN_NUM_INFERENCE_STEPS", "40")))
    max_steps = int(get_env("MAX_STEPS", "50"))
    if steps > max_steps:
        steps = max_steps
    guidance = float((inp or {}).get("guidance") or float(get_env("GEN_GUIDANCE_SCALE", "7.0")))
    max_g = float(get_env("MAX_GUIDANCE", "12.0"))
    if guidance > max_g:
        guidance = max_g
    seed = (inp or {}).get("seed") or get_env("GEN_SEED")
    negative_prompt = (inp or {}).get("negative_prompt") or ""
    use_refiner = (inp or {}).get("use_refiner")
    if use_refiner is None:
        use_refiner = get_env("GEN_ENABLE_REFINER", "false").lower() == "true"
    refiner_steps = int((inp or {}).get("refiner_steps") or int(get_env("GEN_REFINER_STEPS", "20")))
    nsfw = (inp or {}).get("nsfw")
    if nsfw is None:
        nsfw = get_env("NSFW_ENABLED", "true").lower() == "true"
    start = time.time()
    try:
        img_bytes, w, h = generate_sdxl(prompt, negative_prompt, width, height, steps, guidance, seed, use_refiner, refiner_steps)
    except Exception as e:
        logger.error(f"generation_error: {e}")
        w = clamp(int(width), 64, 4096)
        h = clamp(int(height), 64, 4096)
        img = Image.new("RGB", (w, h), (20, 20, 20))
        d = ImageDraw.Draw(img)
        text = (prompt or "NSFW")[:200]
        d.text((20, 20), text, fill=(240, 240, 240))
        b = io.BytesIO()
        img.save(b, format="PNG")
        img_bytes = b.getvalue()
    file_name = f"img_{int(time.time()*1000)}.png"
    url = upload_s3(img_bytes, file_name)
    duration = time.time() - start
    cps = float(get_env("RUNPOD_GPU_COST_PER_SECOND_USD", "0") or "0")
    est_cost = round(duration * cps, 6) if cps > 0 else 0
    meta = {"width": w, "height": h, "steps": steps, "guidance": guidance, "seed": seed, "nsfw": nsfw, "use_refiner": use_refiner, "duration_seconds": round(duration, 3), "estimated_cost_usd": est_cost}
    if url:
        return {"image_url": url, "meta": meta}
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return {"image_data_url": "data:image/png;base64," + b64, "meta": meta}

runpod.serverless.start({"handler": handler})
