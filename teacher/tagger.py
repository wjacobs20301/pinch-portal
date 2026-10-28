"""Describe a photo with Danbooru-style tags (hair colour, clothing, expression, ...) using the WD14 ViT tagger v3
(SmilingWolf/wd-vit-tagger-v3, Apache-2.0). SDXL anime models need these per-image tags to draw well.
Runs on the CPU with onnxruntime.
"""
import csv
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort

REPO = "SmilingWolf/wd-vit-tagger-v3"
# tags that describe the photo as a photo (or are unhelpful) rather than the person/scene
DROP = {"realistic", "photo_(medium)", "photorealistic", "blurry", "blurry_background", "depth_of_field",
        "lips", "nose", "censored", "mosaic_censoring", "watermark", "web_address", "selfie"}


class Tagger:
    def __init__(self, threshold=0.35):
        self.sess = ort.InferenceSession(hf_hub_download(REPO, "model.onnx"), providers=["CPUExecutionProvider"])
        with open(hf_hub_download(REPO, "selected_tags.csv")) as f:
            rows = list(csv.DictReader(f))
        self.names = [r["name"] for r in rows]
        self.general = np.array([r["category"] == "0" for r in rows])    # 0 = general, 4 = character, 9 = rating
        self.threshold = threshold
        self.size = self.sess.get_inputs()[0].shape[1]

    def __call__(self, img: Image.Image, limit=30):
        img = img.convert("RGB")
        s = max(img.size)
        canvas = Image.new("RGB", (s, s), (255, 255, 255)); canvas.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
        x = np.asarray(canvas.resize((self.size, self.size), Image.BICUBIC), np.float32)[:, :, ::-1][None]   # BGR, 0-255
        probs = self.sess.run(None, {self.sess.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0]
        idx = [i for i in np.argsort(-probs) if self.general[i] and probs[i] >= self.threshold and self.names[i] not in DROP]
        return [self.names[i].replace("_", " ") for i in idx[:limit]]
