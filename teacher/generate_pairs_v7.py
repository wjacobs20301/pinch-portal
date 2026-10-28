"""Phase 2 teacher run: full-quality, face-focused training pairs.

Compared with generate_pairs.py (round 3):
  * full quality: full VAE + 25 UniPC steps instead of the LCM / tiny-VAE shortcuts (crisper lines)
  * higher resolution (896 px long side) and one fixed seed for every image (a more consistent teacher → sharper student)
  * --face-detail: for wide shots, the face is redrawn a second time zoomed in (like "ADetailer") and blended back,
    so small faces come out as crisp as close-ups
  * writes NAME_face.txt (face box, 0–1 coords) next to each pair so training can weight faces more

  .venv/bin/python generate_pairs_v7.py --src data/raw --dst data/pairs_v7 --face-detail
  .venv/bin/python generate_pairs_v7.py --src data/raw_faces_full --dst data/faces_v7 --strength 0.75 --cn 0.7 --cn-end 0.5
"""
import argparse, random, time
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image, ImageFilter
from controlnet_aux import LineartDetector
from teacher import load, fit, PROMPT, NEGATIVE

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/raw"); ap.add_argument("--dst", default="data/pairs_v7")
ap.add_argument("--size", type=int, default=896)
ap.add_argument("--steps", type=int, default=25); ap.add_argument("--cfg", type=float, default=7.0)
ap.add_argument("--strength", type=float, default=0.70)
ap.add_argument("--cn", type=float, default=0.8); ap.add_argument("--cn-end", type=float, default=0.6)
ap.add_argument("--face-detail", action="store_true", help="redraw small faces zoomed in and blend them back")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
SEED = 1234

src, dst = Path(a.src), Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
todo = sorted(p for p in src.glob("*.jpg") if not (dst / f"{p.stem}_out.jpg").exists())
random.Random(0).shuffle(todo)
if a.limit: todo = todo[:a.limit]
print(f"{len(todo)} photos to process", flush=True)

pipe = load("stablediffusionapi/counterfeit-v30", tiny_vae=False, lcm=False)
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def draw(img, strength, cn, cn_end):
    ctl = lineart(img, detect_resolution=img.width, image_resolution=img.width).resize(img.size)
    return pipe(PROMPT, negative_prompt=NEGATIVE, image=img, control_image=ctl, strength=strength,
                num_inference_steps=a.steps, controlnet_conditioning_scale=cn, control_guidance_end=cn_end,
                guidance_scale=a.cfg, generator=torch.Generator().manual_seed(SEED)).images[0]


def find_face(img):
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    f = faces.detectMultiScale(g, 1.1, 6, minSize=(max(24, img.width // 40),) * 2)
    return max(f, key=lambda r: r[2] * r[3]) if len(f) else None


def face_detail(photo, anime, box):
    """Redraw the face region of `anime` from the matching photo crop at 640 px, and feather it back in."""
    x, y, w, h = box
    W, H = photo.size
    bw = min(w * 2.2, W); bh = min(bw * 3 / 4, H); bw = bh * 4 / 3
    x0 = int(min(max(x + w / 2 - bw / 2, 0), W - bw)); y0 = int(min(max(y + h * 0.55 - bh / 2, 0), H - bh))
    region = (x0, y0, x0 + int(bw), y0 + int(bh))
    crop = fit(photo.crop(region), 640)
    face = draw(crop, 0.75, 0.7, 0.5).resize((region[2] - region[0], region[3] - region[1]), Image.LANCZOS)
    # soft elliptical mask so the redrawn face blends into the scene without a visible seam
    m = Image.new("L", face.size, 0)
    mw, mh = face.size
    from PIL import ImageDraw
    ImageDraw.Draw(m).ellipse((mw * 0.12, mh * 0.08, mw * 0.88, mh * 0.95), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(mw * 0.06))
    out = anime.copy(); out.paste(face, region[:2], m)
    return out


t0 = time.time()
for i, p in enumerate(todo, 1):
    try:
        img = fit(Image.open(p), a.size)
        out = draw(img, a.strength, a.cn, a.cn_end)
        box = find_face(img)
        if a.face_detail and box is not None and box[2] < 0.45 * img.width:
            out = face_detail(img, out, box)
        img.save(dst / f"{p.stem}_in.jpg", quality=95)
        out.save(dst / f"{p.stem}_out.jpg", quality=95)
        if box is not None:
            x, y, w, h = box
            (dst / f"{p.stem}_face.txt").write_text(f"{x / img.width:.4f} {y / img.height:.4f} {(x + w) / img.width:.4f} {(y + h) / img.height:.4f}")
    except Exception as e:
        print("failed", p.name, repr(e)[:200], flush=True)
    if i % 25 == 0 or i == len(todo):
        el = time.time() - t0
        print(f"{i}/{len(todo)}  {el / i:.1f}s each  ~{el / i * (len(todo) - i) / 3600:.1f}h left", flush=True)
