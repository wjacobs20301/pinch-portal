"""Render style options for the user to pick from: models × redraw strengths on a few webcam frames."""
import sys, time, gc, torch
from PIL import Image
from controlnet_aux import LineartDetector
from teacher import load, fit, PROMPT, NEGATIVE

frames = ["samples/frame1.png", "samples/frame2.png"]
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
for tag, model in [("anything", "stablediffusionapi/anything-v5"), ("counterfeit", "stablediffusionapi/counterfeit-v30")]:
    pipe = load(model, tiny_vae=True, lcm=True)
    for f in frames:
        src = fit(Image.open(f), 640)
        control = lineart(src, detect_resolution=src.width, image_resolution=src.width).resize(src.size)
        for strength in (0.55, 0.75):
            t = time.time()
            out = pipe(PROMPT, negative_prompt=NEGATIVE, image=src, control_image=control, strength=strength,
                       num_inference_steps=8, controlnet_conditioning_scale=0.9, guidance_scale=1.5,
                       generator=torch.Generator().manual_seed(7)).images[0]
            name = f"samples/{tag}_{f.split('/')[-1][:-4]}_s{int(strength*100)}.png"
            out.save(name); print(name, f"{time.time()-t:.1f}s", flush=True)
    del pipe; gc.collect(); torch.mps.empty_cache()
