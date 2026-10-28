"""Make face close-up crops from data/raw → data/raw_faces, for face-focused training.

In the app the finger shape usually frames a face, and the teacher draws much better anime faces when the face
fills the image. For each photo, find the largest face (OpenCV's built-in detector), take a 4:3 box about
2.6x the face width (head, hair and shoulders), and save it at 640x480.
"""
import sys
from pathlib import Path
import cv2

# usage: make_faces.py [src] [dst] [width]   e.g. make_faces.py data/raw_full data/raw_faces_full 896
src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "data/raw_faces")
OUT_W = int(sys.argv[3]) if len(sys.argv) > 3 else 640
dst.mkdir(parents=True, exist_ok=True)
det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
found = 0
for p in sorted(src.glob("*.jpg")):
    img = cv2.imread(str(p)); H, W = img.shape[:2]
    faces = det.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 6, minSize=(48, 48))
    if len(faces) == 0: continue
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    bw = min(w * 2.6, W, H * 4 / 3); bh = bw * 3 / 4
    cx, cy = x + w / 2, y + h * 0.65                      # a little below centre: include shoulders
    x0 = int(min(max(cx - bw / 2, 0), W - bw)); y0 = int(min(max(cy - bh / 2, 0), H - bh))
    crop = img[y0:y0 + int(bh), x0:x0 + int(bw)]
    cv2.imwrite(str(dst / f"face_{p.name}"), cv2.resize(crop, (OUT_W, OUT_W * 3 // 4), interpolation=cv2.INTER_LANCZOS4),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    found += 1
print(f"{found} face crops from {len(list(src.glob('*.jpg')))} photos")
