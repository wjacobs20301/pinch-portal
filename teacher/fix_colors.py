"""Give the teacher's anime drawings the photo's real colors (hair, skin, clothes), keeping the drawing itself.

The teacher sometimes recolors hair (blonde → orange, brown → red) and clothes, which the live model then copies,
seemingly at random. Here each output keeps its own brightness, lines and shading, but its broad color
(Lab a/b, blurred) is replaced by the photo's. With --keep-eyes the eyes keep the teacher's anime colors
(found with MediaPipe face landmarks on the photo). CPU only.

  .venv/bin/python fix_colors.py data/pairs_v7 data/pairs_v7c --keep-eyes
"""
import sys, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage import color
from scipy.ndimage import gaussian_filter

EYES = [[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]]
_landmarker = None


def eye_mask(photo):
    """Soft 0–1 mask over both eyes, generously enlarged because the anime eyes are drawn bigger than the real ones."""
    global _landmarker
    import mediapipe as mp
    if _landmarker is None:
        import urllib.request
        model = Path.home() / ".cache" / "face_landmarker.task"
        if not model.exists():
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", model)
        opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)), num_faces=1)
        _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    res = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(photo.convert("RGB"))))
    if not res.face_landmarks: return None
    W, H = photo.size
    m = Image.new("L", photo.size, 0); d = ImageDraw.Draw(m)
    for ids in EYES:
        pts = np.array([(res.face_landmarks[0][i].x * W, res.face_landmarks[0][i].y * H) for i in ids])
        c = pts.mean(0); pts = c + (pts - c) * 1.9
        d.polygon([tuple(p) for p in pts], fill=255)
    return np.asarray(m.filter(ImageFilter.GaussianBlur(max(2, W / 200))), np.float32) / 255


def keep_photo_colors(photo, anime, sigma=6, amount=0.85, keep=None):
    p = color.rgb2lab(np.asarray(photo, dtype=np.float32) / 255)
    a = color.rgb2lab(np.asarray(anime, dtype=np.float32) / 255)
    w = amount if keep is None else amount * (1 - keep)
    for ch in (1, 2):
        a[..., ch] += w * (gaussian_filter(p[..., ch], sigma) - gaussian_filter(a[..., ch], sigma))
    return Image.fromarray((np.clip(color.lab2rgb(a), 0, 1) * 255).astype(np.uint8))


def one(args):
    p_in, dst, keep_eyes = args
    p_out = Path(str(p_in).replace("_in.jpg", "_out.jpg"))
    d_in, d_out = dst / p_in.name, dst / p_out.name
    if d_out.exists(): return
    photo = Image.open(p_in).convert("RGB")
    anime = Image.open(p_out).convert("RGB").resize(photo.size)
    keep_photo_colors(photo, anime, keep=eye_mask(photo) if keep_eyes else None).save(d_out, quality=95)
    if not d_in.exists(): os.link(p_in, d_in)          # same input photo, no extra disk space
    side = Path(str(p_in).replace("_in.jpg", "_face.txt"))
    if side.exists() and not (dst / side.name).exists(): os.link(side, dst / side.name)


if __name__ == "__main__":
    SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
    keep_eyes = "--keep-eyes" in sys.argv
    DST.mkdir(parents=True, exist_ok=True)
    todo = [(p, DST, keep_eyes) for p in sorted(SRC.glob("*_in.jpg"))]
    with Pool(6) as pool:                               # leave cores for other jobs
        for i, _ in enumerate(pool.imap_unordered(one, todo, chunksize=16), 1):
            if i % 1000 == 0: print(f"{i}/{len(todo)}", flush=True)
    print(f"done: {len(todo)} pairs → {DST}")
