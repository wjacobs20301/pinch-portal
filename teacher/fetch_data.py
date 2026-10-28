"""Download a diverse subset of HaGRIDv2 (people in front of webcams, hands up) without fetching the full 1.5 TB.

Uses HTTP range requests (remotezip) to pull only the chosen images out of each huge per-gesture zip.
Images are resized to 768px on the long side. See data/CREDITS.md for the license.
"""
import io, random, sys, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from remotezip import RemoteZip

BASE = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagrid_v2_zip/"
V1_BASE = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/"
V1_GESTURES = {"peace", "palm", "ok"}   # these older classes live under the v1 path
GESTURES = ["thumb_index", "thumb_index2", "take_picture", "hand_heart", "hand_heart2", "peace", "palm", "ok", "one"]
PER_GESTURE = 250
OUT = Path("data/raw"); OUT.mkdir(parents=True, exist_ok=True)
random.seed(0)


def save(data, name):
    im = Image.open(io.BytesIO(data)).convert("RGB")
    s = 768 / max(im.size)
    im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS).save(OUT / name, quality=92)


def grab_from(gesture):
    url = (V1_BASE if gesture in V1_GESTURES else BASE) + gesture + ".zip"
    with RemoteZip(url) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
        pick = random.sample(names, min(PER_GESTURE, len(names)))
    def one(n):
        dest = f"{gesture}_{Path(n).stem}.jpg"
        if (OUT / dest).exists(): return
        for attempt in range(3):
            try:
                with RemoteZip(url) as z: save(z.read(n), dest); return
            except Exception as e: err = e
        print("failed", n, err, flush=True)
    with ThreadPoolExecutor(8) as ex: list(ex.map(one, pick))
    print(f"{gesture}: done", flush=True)


# 1) the small no_gesture zip, in full
zpath = Path("data/no_gesture.zip")
if any(OUT.glob("no_gesture_*.jpg")): pass
elif not zpath.exists():
    urllib.request.urlretrieve(BASE + "no_gesture.zip", zpath)
if zpath.exists():
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.lower().endswith(".jpg"): save(z.read(n), f"no_gesture_{Path(n).stem}.jpg")
    zpath.unlink()
print("no_gesture: done", flush=True)

# 2) a random slice of each gesture zip, via range requests
only = sys.argv[1:]   # optionally: fetch_data.py peace palm ok
for g in (only or GESTURES):
    try: grab_from(g)
    except Exception as e: print(g, "failed:", e, flush=True)
print("total images:", len(list(OUT.glob("*.jpg"))))
