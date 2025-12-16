# RunPod SDXL Worker

Serverless worker for photorealistic SDXL image generation. Designed for RunPod Serverless Endpoints.

## Contents
- `rp_handler.py` — worker entry, defines `handler(event)` and starts serverless loop
- `requirements.txt` — minimal dependencies for SDXL
- `Dockerfile` — GPU-enabled image build for RunPod (CUDA 12.1)

## Environment Variables
- `GEN_MODEL_ID` (default: `stabilityai/stable-diffusion-xl-base-1.0`)
- `GEN_IMAGE_WIDTH` / `GEN_IMAGE_HEIGHT` (default: `1920` / `1080`)
- `GEN_NUM_INFERENCE_STEPS` (default: `40`)
- `GEN_GUIDANCE_SCALE` (default: `7.0`)
- `GEN_ENABLE_REFINER` (default: `false`)
- `GEN_REFINER_MODEL_ID` (default: `stabilityai/stable-diffusion-xl-refiner-1.0`)
- `GEN_REFINER_STEPS` (default: `20`)
- `HF_TOKEN` (optional, if model requires authentication)
- `MAX_IMAGE_PIXELS` (default: `2073600` for FHD)
- `MAX_STEPS` / `MAX_GUIDANCE` safety caps
- `SAVE_OUTPUTS_TO_S3` (`true` / `false`)
- `S3_ENDPOINT`, `S3_BUCKET`, `S3_REGION`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_PUBLIC_BASE_URL`, `S3_KEY_PREFIX`, `S3_REQUIRE_ACL`, `S3_CACHE_CONTROL`, `S3_FORCE_PATH_STYLE`
- `RUNPOD_GPU_COST_PER_SECOND_USD` (optional — adds estimated cost to metadata)
- `LOG_LEVEL` (`debug|info|warning|error`)
- `SENTRY_DSN` (optional)

## Request/Response Contract
POST `https://api.runpod.io/v2/{ENDPOINT_ID}/runsync`
```json
{
  "input": {
    "prompt": "portrait of a woman, studio soft light, photorealistic",
    "negative_prompt": "lowres, bad anatomy, bad hands, artifacts",
    "width": 1920,
    "height": 1080,
    "steps": 40,
    "guidance": 7.0,
    "seed": null,
    "nsfw": true,
    "use_refiner": false,
    "refiner_steps": 20
  }
}
```
Response:
```json
{
  "output": {
    "image_url": "https://.../images/img_1734360000000.png",
    "meta": { "width": 1920, "height": 1080, "steps": 40, "guidance": 7.0 }
  },
  "status": "SUCCESS"
}
```
Fallback when S3 is not configured:
```json
{
  "output": {
    "image_data_url": "data:image/png;base64,...",
    "meta": { ... }
  },
  "status": "SUCCESS"
}
```

## Deploy on RunPod (GitHub Import)
1. Push this folder as the root of a public GitHub repository (e.g., `runpod-ultra-realistic-sdxl`).
2. In RunPod Console → Serverless → Create Endpoint → Import GitHub Repository → select the repo.
3. Set environment variables as above (S3 or return base64).
4. Choose GPU (e.g., `RTX_3090_24GB`) and set timeout/limits matching your usage.

## Local Smoke Test
Use `scripts/runpod_smoke.py` from the orchestrator repository to test endpoint via API.
