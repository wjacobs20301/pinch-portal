"""Batch-run the anime teacher over data/raw/*.jpg → training pairs in data/pairs/.

For each photo NAME.jpg it writes NAME_in.jpg (the photo, resized) and NAME_out.jpg (the anime redraw), same size.
Resumable: photos that already have an _out.jpg are skipped, so it can be stopped and restarted any time.

  .venv/bin/python generate_pairs.py [--model stablediffusionapi/counterfeit-v30] [--strength 0.55] [--limit N]
"""
import argparse, random, time
from pathlib import Path
import torch
from PIL import Image
from controlnet_aux import LineartDetector
from teacher import load, fit, PROMPT, NEGATIVE

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="stablediffusionapi/counterfeit-v30")
ap.add_argument("--strength", type=float, default=0.55)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--size", type=int, default=640)
ap.add_argument("--limit", type=int, default=0, help="only process this many (0 = all)")
ap.add_argument("--cfg", type=float, default=2.2, help="prompt strength; higher makes the negative prompt count more")
ap.add_argument("--cn", type=float, default=0.9, help="how strongly to follow the photo's outlines")
ap.add_argument("--cn-end", type=float, default=1.0,
                help="stop following outlines after this fraction of steps (<1 frees the teacher to draw anime proportions)")
ap.add_argument("--prompt-extra", default="", help="appended to the teacher prompt")
ap.add_argument("--neg-extra", default="", help="appended to the negative prompt")
ap.add_argument("--src", default="data/raw")
ap.add_argument("--dst", default="data/pairs")
a = ap.parse_args()

src, dst = Path(a.src), Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)
todo = sorted(p for p in src.glob("*.jpg") if not (dst / f"{p.stem}_out.jpg").exists())
random.Random(0).shuffle(todo)            # mix gestures so a partial run is still diverse
if a.limit: todo = todo[:a.limit]
print(f"{len(todo)} photos to process", flush=True)

pipe = load(a.model, tiny_vae=True, lcm=True)
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
t0 = time.time()
for i, p in enumerate(todo, 1):
    try:
        img = fit(Image.open(p), a.size)
        control = lineart(img, detect_resolution=img.width, image_resolution=img.width).resize(img.size)
        out = pipe(PROMPT + a.prompt_extra, negative_prompt=NEGATIVE + a.neg_extra, image=img, control_image=control, strength=a.strength,
                   num_inference_steps=a.steps, controlnet_conditioning_scale=a.cn, control_guidance_end=a.cn_end, guidance_scale=a.cfg,
                   generator=torch.Generator().manual_seed(i)).images[0]
        img.save(dst / f"{p.stem}_in.jpg", quality=93)
        out.save(dst / f"{p.stem}_out.jpg", quality=93)
    except Exception as e:
        print("failed", p.name, e, flush=True)
    if i % 25 == 0 or i == len(todo):
        el = time.time() - t0
        print(f"{i}/{len(todo)}  {el/i:.1f}s each  ~{el/i*(len(todo)-i)/3600:.1f}h left", flush=True)
