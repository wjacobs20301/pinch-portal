"""Re-download the full-resolution (1080p) originals of the photos that contain a usable face, so face close-ups can
be cut from sharp sources instead of the 768px copies in data/raw. Uses HTTP range requests (remotezip), like
fetch_data.py. Output: data/raw_full/<same name>.jpg (no resizing). See data/CREDITS.md for the license.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from remotezip import RemoteZip
BASE = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagrid_v2_zip/"      # as in fetch_data.py
V1_BASE = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/"
V1_GESTURES = {"peace", "palm", "ok"}

OUT = Path("data/raw_full"); OUT.mkdir(parents=True, exist_ok=True)
wanted = sorted(p.name[len("face_"):] for p in Path("data/raw_faces").glob("face_*.jpg"))   # photos with a face
by_gesture = {}
for name in wanted:
    gesture, stem = name.rsplit("_", 1)[0], Path(name).stem.rsplit("_", 1)[1]
    by_gesture.setdefault(gesture, []).append((name, stem))

for gesture, items in by_gesture.items():
    todo = [(n, s) for n, s in items if not (OUT / n).exists()]
    if not todo: continue
    if gesture == "no_gesture":
        url = BASE + "no_gesture.zip"
    else:
        url = (V1_BASE if gesture in V1_GESTURES else BASE) + gesture + ".zip"
    with RemoteZip(url) as z:
        members = {Path(m).stem: m for m in z.namelist() if m.lower().endswith(".jpg")}
    def one(item):
        name, stem = item
        if stem not in members: return f"missing {name}"
        for _ in range(3):
            try:
                with RemoteZip(url) as z: (OUT / name).write_bytes(z.read(members[stem])); return None
            except Exception as e: err = e
        return f"failed {name}: {err}"
    with ThreadPoolExecutor(8) as ex:
        problems = [r for r in ex.map(one, todo) if r]
    print(f"{gesture}: {len(todo) - len(problems)}/{len(todo)} downloaded", *problems[:3], flush=True)
print("total:", len(list(OUT.glob("*.jpg"))))
