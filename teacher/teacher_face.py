"""Expression-preserving anime teacher for face close-ups.

The plain teacher (lineart ControlNet only, released early so it can draw anime proportions) changes expressions at
random — closed mouths become screams, neutral faces frown — so the live model learns mush for mouths.
This adds a second ControlNet driven by MediaPipe face landmarks (CrucibleAI/ControlNetMediaPipeFace, SD1.5),
held for the whole generation, which pins mouth shape, eye openness, gaze and brows to the photo while the
lineart guide is still released early for anime proportions.

  .venv/bin/python teacher_face.py --src data/raw_faces_full --dst data/faces_v9
"""
import argparse, random, time
from pathlib import Path
import torch
from PIL import Image
from controlnet_aux import LineartDetector, MediapipeFaceDetector
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
from teacher import fit, PROMPT, NEGATIVE, device

SEED = 1234


def load():
    dt = torch.float16
    cns = [ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=dt),
           ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", subfolder="diffusion_sd15",
                                           torch_dtype=dt, variant="fp16")]
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("stablediffusionapi/counterfeit-v30", controlnet=cns,
                                                                   torch_dtype=dt, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device())


# The base anime model gives everyone worried brows and grimaces. With the landmark ControlNet pinning the real mouth
# shape (smiles, talking) these negatives only remove the bias instead of erasing real expressions.
EXPR_NEG = ", frown, furrowed brow, worried, sad, grimace, crying, tears, scowl"


# Eye + eyebrow regions (MediaPipe face mesh) — blanked out of the landmark control image in --mouth-only mode,
# so the landmark lock keeps the real mouth/jaw/nose while the anime model is free to draw big anime eyes.
EYE_BROW = [[70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 33, 7, 163, 144, 145, 153, 154, 155, 133],
            [300, 293, 334, 296, 336, 285, 295, 282, 283, 276, 263, 249, 390, 373, 374, 380, 381, 382, 362]]


def blank_eyes(mesh, img):
    """Black out (= no guidance) the eye and brow areas of the landmark control image."""
    from align_faces import landmarks
    from PIL import ImageDraw
    pts = landmarks(img)
    if pts is None: return mesh
    out = mesh.copy(); d = ImageDraw.Draw(out)
    for ids in EYE_BROW:
        q = pts[ids]; c = q.mean(0); q = c + (q - c) * 1.6                       # generous margin around eye + brow
        x0, y0 = q.min(0); x1, y1 = q.max(0)
        d.ellipse([x0, y0, x1, y1], fill=(0, 0, 0))
    return out


class FaceTeacher:
    def __init__(self, strength=0.75, line=0.7, line_end=0.5, face=0.9, steps=25, cfg=7.0, neg=EXPR_NEG, mouth_only=False):
        self.pipe, self.neg, self.mouth_only = load(), neg, mouth_only
        self.cfg = dict(strength=strength, line=line, line_end=line_end, face=face, steps=steps, cfg=cfg)
        self.lineart, self.mesh = LineartDetector.from_pretrained("lllyasviel/Annotators"), MediapipeFaceDetector()

    def __call__(self, img):
        c = self.cfg
        line = self.lineart(img, detect_resolution=img.width, image_resolution=img.width).resize(img.size)
        mesh = self.mesh(img, max_faces=1, min_confidence=0.5)
        mesh = mesh.resize(img.size) if mesh is not None else Image.new("RGB", img.size)
        if self.mouth_only: mesh = blank_eyes(mesh, img)
        return self.pipe(PROMPT, negative_prompt=NEGATIVE + self.neg, image=img, control_image=[line, mesh], strength=c["strength"],
                         num_inference_steps=c["steps"], controlnet_conditioning_scale=[c["line"], c["face"]],
                         control_guidance_start=[0.0, 0.0], control_guidance_end=[c["line_end"], 1.0],
                         guidance_scale=c["cfg"], generator=torch.Generator().manual_seed(SEED)).images[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw_faces_full"); ap.add_argument("--dst", default="data/faces_v9")
    ap.add_argument("--size", type=int, default=896); ap.add_argument("--face", type=float, default=0.9)
    ap.add_argument("--no-expr-neg", action="store_true")
    ap.add_argument("--mouth-only", action="store_true", help="lock mouth/jaw/nose only; let the model draw anime eyes")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--names", default="", help="file with names to do")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    todo = [src / n.strip() for n in Path(a.names).read_text().split()] if a.names else sorted(src.glob("*.jpg"))
    todo = [p for p in todo if not (dst / f"{p.stem}_out.jpg").exists()]
    random.Random(0).shuffle(todo)
    if a.limit: todo = todo[:a.limit]
    print(f"{len(todo)} photos to process", flush=True)
    t, t0 = FaceTeacher(face=a.face, neg="" if a.no_expr_neg else EXPR_NEG, mouth_only=a.mouth_only), time.time()
    for i, p in enumerate(todo, 1):
        try:
            img = fit(Image.open(p), a.size)
            out = t(img)
            img.save(dst / f"{p.stem}_in.jpg", quality=95); out.save(dst / f"{p.stem}_out.jpg", quality=95)
        except Exception as e:
            print("failed", p.name, repr(e)[:200], flush=True)
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"{i}/{len(todo)}  {el / i:.1f}s each  ~{el / i * (len(todo) - i) / 3600:.1f}h left", flush=True)
