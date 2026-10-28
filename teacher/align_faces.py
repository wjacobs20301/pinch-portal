"""Build the face model's training data: upright, aligned 256x256 face crops (photo → anime), with clean lighting.

For each teacher pair (photo + anime drawing):
  1. colours: hair/skin/clothes take the photo's real colours, eyes keep the anime colours (fix_colors)
  2. lighting: the teacher draws dramatic, dark shadows across faces; the drawing's broad light/dark areas are pulled
     toward the photo's (usually even) lighting and lifted slightly, keeping all lines and detail (high frequencies)
  3. alignment: MediaPipe landmarks on the photo → similarity transform putting the eyes and mouth at fixed spots
     (same transform as the live app uses), applied to both images
Writes data/face_aligned/NAME_in.jpg / NAME_out.jpg (CROP x CROP, a little larger than 256 so training can jitter).

  .venv/bin/python align_faces.py data/faces_v7 data/pairs_v7 --out data/face_aligned
"""
import sys, argparse
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import color
from scipy.ndimage import gaussian_filter
import cv2
from fix_colors import keep_photo_colors, eye_mask

CROP = 288                                    # saved size; the model sees random 256 crops of it
# where the features go in a 256x256 face crop (x, y): eye centres and mouth centre — matched in the app (index.html)
ALIGN = np.float32([[0.35 * 256, 0.40 * 256], [0.65 * 256, 0.40 * 256], [0.50 * 256, 0.73 * 256]]) + (CROP - 256) / 2
EYE_L, EYE_R, MOUTH = [33, 133], [362, 263], [13, 14, 61, 291]
_lm = None


def landmarks(img):
    global _lm
    import mediapipe as mp
    if _lm is None:
        model = Path.home() / ".cache" / "face_landmarker.task"
        _lm = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)), num_faces=1))
    r = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img)))
    if not r.face_landmarks: return None
    W, H = img.size
    return np.float32([(p.x * W, p.y * H) for p in r.face_landmarks[0]])


def relight(photo, anime, sigma=12, amount=0.55, lift=0.92):
    """Pull the drawing's broad lighting toward the photo's and lift dark tones; lines/details are untouched."""
    p = color.rgb2lab(np.asarray(photo, np.float32) / 255); a = color.rgb2lab(np.asarray(anime, np.float32) / 255)
    a[..., 0] += amount * (gaussian_filter(p[..., 0], sigma) - gaussian_filter(a[..., 0], sigma))
    a[..., 0] = 100 * np.clip(a[..., 0] / 100, 0, 1) ** lift
    return Image.fromarray((np.clip(color.lab2rgb(a), 0, 1) * 255).astype(np.uint8))


def one(args):
    p_in, out = args
    name = p_in.name[: -len("_in.jpg")]
    if (out / f"{name}_out.jpg").exists(): return 0
    photo = Image.open(p_in).convert("RGB")
    anime = Image.open(str(p_in)[: -len("_in.jpg")] + "_out.jpg").convert("RGB").resize(photo.size)
    pts = landmarks(photo)
    if pts is None: return 0
    src = np.float32([pts[EYE_L].mean(0), pts[EYE_R].mean(0), pts[MOUTH].mean(0)])
    if np.linalg.norm(src[1] - src[0]) < 40: return 0                # face too small in this photo to be useful
    anime = relight(photo, keep_photo_colors(photo, anime, keep=eye_mask(photo)))
    M, _ = cv2.estimateAffinePartial2D(src, ALIGN)                  # rotation + uniform scale + shift
    warp = lambda im: Image.fromarray(cv2.warpAffine(np.asarray(im), M, (CROP, CROP), flags=cv2.INTER_LANCZOS4,
                                                     borderMode=cv2.BORDER_REFLECT))
    warp(photo).save(out / f"{name}_in.jpg", quality=95); warp(anime).save(out / f"{name}_out.jpg", quality=95)
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", default="data/face_aligned")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = [p for d in a.dirs for p in sorted(Path(d).glob("*_in.jpg"))]
    if a.limit: files = files[:: max(1, len(files) // a.limit)][: a.limit]
    with Pool(6) as pool:
        n = sum(pool.imap_unordered(one, [(p, out) for p in files], chunksize=8))
    print(f"aligned faces: {n} new from {len(files)} pairs → {out}")
