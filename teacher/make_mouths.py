"""Find the mouth in every training pair's photo (MediaPipe face landmarks, lip outline) and save its box as
NAME_mouth.txt (0–1 coords, padded) so training can give mouths extra weight. CPU only.

  .venv/bin/python make_mouths.py data/pairs_v7c data/faces_v7c
"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
_lm = None


def landmarker():
    global _lm
    if _lm is None:
        import mediapipe as mp, urllib.request
        model = Path.home() / ".cache" / "face_landmarker.task"
        if not model.exists():
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", model)
        _lm = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)), num_faces=1))
    return _lm


def one(p):
    dst = Path(str(p)[: -len("_in.jpg")] + "_mouth.txt")
    if dst.exists(): return 0
    import mediapipe as mp
    img = np.asarray(Image.open(p).convert("RGB"))
    res = landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    if not res.face_landmarks: return 0
    pts = np.array([(res.face_landmarks[0][i].x, res.face_landmarks[0][i].y) for i in LIPS])
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    px, py = (x1 - x0) * 0.35, (y1 - y0) * 0.6 + (x1 - x0) * 0.1     # include smile lines and chin crease
    dst.write_text(f"{max(0, x0 - px):.4f} {max(0, y0 - py):.4f} {min(1, x1 + px):.4f} {min(1, y1 + py):.4f}")
    return 1


if __name__ == "__main__":
    files = [p for d in sys.argv[1:] for p in sorted(Path(d).glob("*_in.jpg"))]
    with Pool(6) as pool:
        found = sum(pool.imap_unordered(one, files, chunksize=32))
    print(f"mouths found in {found} of {len(files)} images (others: no face, or already done)")
