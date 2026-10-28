"""Anime "teacher": redraws a webcam frame as anime with Stable Diffusion + ControlNet lineart.

Usage: .venv/bin/python teacher.py INPUT.png OUTPUT.png [--model ...] [--strength 0.6] [--steps 20]
The slow, high-quality output of this script is what the small live model will learn to imitate.
"""
import argparse, time
import torch
from PIL import Image
from controlnet_aux import LineartDetector
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, AutoencoderTiny, UniPCMultistepScheduler, LCMScheduler

# Generic on purpose: the live model must work for anyone, so colors and layout come from the photo itself.
PROMPT = ("anime screencap, clean lineart, flat cel shading, vibrant colors, big expressive anime eyes, "
          "person looking at viewer, natural hair color, indoors, room interior, plain wall, masterpiece, best quality")
# Speed lines / dyed hair vary frame to frame, which would make the live model flicker — keep them out.
NEGATIVE = ("photo, realistic, 3d, blurry, lowres, bad anatomy, extra fingers, deformed, text, watermark, "
            "speed lines, emphasis lines, motion lines, abstract background, blue hair, aqua hair, dyed hair, multicolored hair")


def device():
    """CUDA (NVIDIA, or AMD via ROCm builds of PyTorch) on the Linux desktop, Apple GPU on the Mac, else CPU."""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load(model, lineart_cn="lllyasviel/control_v11p_sd15_lineart", tiny_vae=False, lcm=False):
    dtype = torch.float16
    cn = ControlNetModel.from_pretrained(lineart_cn, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model, controlnet=cn, torch_dtype=dtype, safety_checker=None)
    if tiny_vae:
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype)
    if lcm:   # LCM-LoRA: good results in ~4 steps instead of ~20
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device())


def fit(img, long_side=768):
    w, h = img.size
    s = long_side / max(w, h)
    return img.convert("RGB").resize((round(w * s / 8) * 8, round(h * s / 8) * 8), Image.LANCZOS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--model", default="stablediffusionapi/anything-v5")
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cn", type=float, default=0.9)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lcm", action="store_true")
    ap.add_argument("--tiny-vae", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="run N times to measure steady-state speed")
    a = ap.parse_args()

    t0 = time.time()
    pipe = load(a.model, tiny_vae=a.tiny_vae, lcm=a.lcm)
    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
    print(f"loaded in {time.time() - t0:.0f}s")

    src = fit(Image.open(a.input), a.size)
    for r in range(a.repeat):
        t1 = time.time()
        control = lineart(src, detect_resolution=src.width, image_resolution=src.width).resize(src.size)
        out = pipe(PROMPT, negative_prompt=NEGATIVE, image=src, control_image=control, strength=a.strength,
                   num_inference_steps=a.steps, controlnet_conditioning_scale=a.cn, guidance_scale=1.5 if a.lcm else 7,
                   generator=torch.Generator().manual_seed(a.seed)).images[0]
        print(f"generated {src.size} in {time.time() - t1:.1f}s", flush=True)
    out.save(a.output)
    control.save(a.output.replace(".png", "_lineart.png"))
