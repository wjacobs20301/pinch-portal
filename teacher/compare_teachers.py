"""Phase 1: compare candidate anime teachers on a fixed test set, for quality AND consistency.

  A  current teacher: SD1.5 Counterfeit + lineart ControlNet, LCM 8 steps, tiny VAE (fast but soft)
  B  same model at full quality: full VAE, 25 UniPC steps
  C  SDXL Animagine XL 4.0 + xinsir anime-painter ControlNet (outlines → anime), 28 steps

Consistency: every image is also run shifted by 3 px with the SAME seed; a good teacher draws both the same
(inconsistent teachers are what make the live model blurry). Writes samples/teachers/<id>/ and a summary.

  .venv/bin/python compare_teachers.py samples/teacher_test
"""
import sys, time, json, gc
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from controlnet_aux import LineartDetector
from teacher import load, fit, PROMPT, NEGATIVE, device

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/teacher_test")
OUT = Path("samples/teachers")
SEED = 1234                                    # fixed seed: same input → same drawing
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")

XL_PROMPT = ("anime screencap, modern anime, clean lineart, cel shading, flat colors, detailed eyes, "
             "looking at viewer, indoors, masterpiece, high score, great score, absurdres")
XL_NEG = ("lowres, bad anatomy, bad hands, extra digits, missing finger, text, watermark, signature, worst quality, "
          "low quality, low score, bad score, average score, blurry, photo, realistic, 3d, speed lines, "
          "angry, furrowed brow, multicolored hair")


def shifted(img, dx=3, dy=2):
    w, h = img.size
    return img.crop((dx, dy, w, h)).resize((w, h), Image.LANCZOS)


def sd15(full_quality):
    pipe = load("stablediffusionapi/counterfeit-v30", tiny_vae=not full_quality, lcm=not full_quality)
    steps, cfg = (25, 7.0) if full_quality else (8, 2.2)
    def run(img, size):
        img = fit(img, size)
        ctl = lineart(img, detect_resolution=img.width, image_resolution=img.width).resize(img.size)
        return pipe(PROMPT, negative_prompt=NEGATIVE, image=img, control_image=ctl, strength=0.72,
                    num_inference_steps=steps, controlnet_conditioning_scale=0.8, control_guidance_end=0.6,
                    guidance_scale=cfg, generator=torch.Generator().manual_seed(SEED)).images[0]
    return pipe, run, 640


def sdxl():
    """Animagine XL 4.0 + anime-painter ControlNet, fitted into an 8 GB GPU: 8-bit UNet (~2.7 GB) + ControlNet (2.5 GB)
    + VAE on the GPU; the fixed prompt is encoded once on the CPU and the text encoders are dropped (~11 s/image)."""
    from diffusers import (ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL,
                           UNet2DConditionModel, BitsAndBytesConfig)
    dt, dev = torch.float16, device()
    q = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)} if dev == "cuda" else {}
    unet = UNet2DConditionModel.from_pretrained("cagliostrolab/animagine-xl-4.0", subfolder="unet", torch_dtype=dt, **q)
    cn = ControlNetModel.from_pretrained("xinsir/anime-painter", torch_dtype=dt).to(dev)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dt).to(dev)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-4.0", controlnet=cn, vae=vae, unet=unet, torch_dtype=dt)
    if dev != "cuda": pipe.unet.to(dev)
    pipe.set_progress_bar_config(disable=True); pipe.vae.enable_tiling()
    pe, npe, ppe, nppe = pipe.encode_prompt(XL_PROMPT, device="cpu", num_images_per_prompt=1,
                                            do_classifier_free_guidance=True, negative_prompt=XL_NEG)
    pipe.text_encoder = pipe.text_encoder_2 = None
    emb = dict(prompt_embeds=pe.to(dev), negative_prompt_embeds=npe.to(dev),
               pooled_prompt_embeds=ppe.to(dev), negative_pooled_prompt_embeds=nppe.to(dev))
    def run(img, size):
        img = fit(img, size)
        ctl = lineart(img, detect_resolution=img.width, image_resolution=img.width).resize(img.size)
        return pipe(**emb, image=img, control_image=ctl, strength=0.7,
                    num_inference_steps=28, controlnet_conditioning_scale=0.8, control_guidance_end=0.7,
                    guidance_scale=6.0, generator=torch.Generator().manual_seed(SEED)).images[0]
    return pipe, run, 896


def sdxl_tagged():
    """D: Animagine XL 4.0 drawing FROM SCRATCH (txt2img) guided by the photo's outlines (anime-painter ControlNet)
    and per-image WD14 tags; colours are restored from the photo afterwards (fix_colors)."""
    from diffusers import (ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL,
                           UNet2DConditionModel, BitsAndBytesConfig)
    from tagger import Tagger
    from fix_colors import keep_photo_colors
    dt, dev = torch.float16, device()
    q = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)} if dev == "cuda" else {}
    unet = UNet2DConditionModel.from_pretrained("cagliostrolab/animagine-xl-4.0", subfolder="unet", torch_dtype=dt, **q)
    cn = ControlNetModel.from_pretrained("xinsir/anime-painter", torch_dtype=dt).to(dev)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dt).to(dev)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-4.0", controlnet=cn, vae=vae, unet=unet, torch_dtype=dt)
    if dev != "cuda": pipe.unet.to(dev)
    te = (pipe.text_encoder.to("cpu", torch.float32), pipe.text_encoder_2.to("cpu", torch.float32))   # per-image prompts, CPU
    pipe.text_encoder = pipe.text_encoder_2 = None        # attached only while encoding, so drawing stays on GPU/fp16
    pipe.set_progress_bar_config(disable=True); pipe.vae.enable_tiling()
    tag = Tagger()
    def run(img, size):
        img = fit(img, size)
        tags = tag(img)
        prompt = ", ".join(tags) + ", anime screencap, modern anime, clean lineart, cel shading, " \
                 "masterpiece, high score, great score, absurdres"
        with torch.no_grad():
            pipe.text_encoder, pipe.text_encoder_2 = te
            pe, npe, ppe, nppe = pipe.encode_prompt(prompt, device="cpu", num_images_per_prompt=1,
                                                    do_classifier_free_guidance=True, negative_prompt=XL_NEG)
            pipe.text_encoder = pipe.text_encoder_2 = None
        ctl = lineart(img, detect_resolution=img.width, image_resolution=img.width).resize(img.size)
        out = pipe(prompt_embeds=pe.to(dev, dt), negative_prompt_embeds=npe.to(dev, dt),
                   pooled_prompt_embeds=ppe.to(dev, dt), negative_pooled_prompt_embeds=nppe.to(dev, dt),
                   image=ctl, width=img.width, height=img.height, num_inference_steps=28,
                   controlnet_conditioning_scale=0.9, guidance_scale=6.0,
                   generator=torch.Generator().manual_seed(SEED)).images[0]
        run.last_tags = tags
        return keep_photo_colors(img, out)
    return pipe, run, 896


if __name__ == "__main__":
    which = sys.argv[2].split(",") if len(sys.argv) > 2 else ["A", "B", "C"]
    imgs = sorted(SRC.glob("*.jpg")) + sorted(SRC.glob("*.png"))
    summary = {}
    for tid in which:
        pipe, run, size = {"A": lambda: sd15(False), "B": lambda: sd15(True), "C": sdxl, "D": sdxl_tagged}[tid]()
        d = OUT / tid; d.mkdir(parents=True, exist_ok=True)
        t0, diffs = time.time(), []
        for p in imgs:
            im = Image.open(p).convert("RGB")
            a = run(im, size); a.save(d / f"{p.stem}.jpg", quality=93)
            if hasattr(run, "last_tags"): print(p.stem[:30], "→", ", ".join(run.last_tags[:12]), flush=True)
            b = run(shifted(im), size)
            # undo the shift (scaled to output size) before comparing
            sx, sy = round(3 * a.width / im.width), round(2 * a.height / im.height)
            A = np.asarray(a, np.float32)[sy:, sx:]; B = np.asarray(b.resize(a.size), np.float32)[: a.height - sy, : a.width - sx]
            diffs.append(float(np.abs(A - B).mean() / 255))
        summary[tid] = {"sec_per_image": round((time.time() - t0) / (2 * len(imgs)), 1),
                        "inconsistency": round(float(np.mean(diffs)), 4)}
        print(tid, summary[tid], flush=True)
        del pipe; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    prev = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    (OUT / "summary.json").write_text(json.dumps(prev | summary, indent=1))
