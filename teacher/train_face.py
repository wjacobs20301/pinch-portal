"""Train the dedicated face model (FaceStudent): aligned 256x256 photo face → anime face.

Data: data/face_aligned_v9/NAME_in.jpg / NAME_out.jpg (288x288, from align_faces.py; eyes and mouth at fixed spots),
so the model sees random 256 crops with a few pixels of jitter, like the live app's alignment wobble.
Because faces are aligned, eyes and mouth are always in the same place: a fixed weight map makes the mouth count 6x
and the eyes 3x in every loss. GAN kept gentle (the last run's strong critics made faces harsh and gritty).

  .venv/bin/python train_face.py --data data/face_aligned_v9
"""
import argparse, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from student import FaceStudent
from losses import VGG, Critic, edges, jitter, hinge_d

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="face_v9"); ap.add_argument("--data", default="data/face_aligned_v9")
ap.add_argument("--width", type=int, default=24); ap.add_argument("--iters", type=int, default=60000)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--gan-weight", type=float, default=0.03); ap.add_argument("--gan-start", type=int, default=5000)
ap.add_argument("--temporal", type=float, default=3.0); ap.add_argument("--line", type=float, default=1.0)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--init", default="", help="start from another face checkpoint's weights (e.g. checkpoints/face_v9.pt)")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class AlignedFaces(Dataset):
    def __init__(self, root):
        self.ins = sorted(Path(root).glob("*_in.jpg"))

    def __len__(self): return 10 ** 7

    def __getitem__(self, k):
        p = self.ins[random.randrange(len(self.ins))]
        x = Image.open(p).convert("RGB"); y = Image.open(str(p)[: -len("_in.jpg")] + "_out.jpg").convert("RGB")
        i, j = random.randint(8, 24), random.randint(8, 24)            # ±8 px wobble around the aligned centre
        x, y = TF.crop(x, i, j, 256, 256), TF.crop(y, i, j, 256, 256)
        if random.random() < 0.5: x, y = TF.hflip(x), TF.hflip(y)
        x, y = TF.to_tensor(x) * 2 - 1, TF.to_tensor(y) * 2 - 1
        x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)).clamp(-1, 1)
        x = (x + torch.randn_like(x) * random.uniform(0, 0.025)).clamp(-1, 1)
        return x, y


def feature_weights(size=256):
    """Fixed weight map for aligned faces: mouth 6x, eyes 3x, rest of face 2x, background 1x (soft-edged)."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    blob = lambda cx, cy, rx, ry: torch.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
    face = blob(0.5, 0.55, 0.32, 0.42)
    eyes = blob(0.35, 0.40, 0.09, 0.06) + blob(0.65, 0.40, 0.09, 0.06)
    mouth = blob(0.5, 0.73, 0.14, 0.08)
    return (1 + face + 2 * eyes.clamp(max=1) + 5 * mouth)[None, None]


if __name__ == "__main__":
    ds = AlignedFaces(a.data)
    print(f"aligned face pairs: {len(ds.ins)}", flush=True)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=6, persistent_workers=True)
    model = FaceStudent(a.width).to(dev)
    critic = Critic().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.5, 0.999))
    dopt = torch.optim.Adam(critic.parameters(), lr=a.lr, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    vgg, W = VGG().to(dev), feature_weights().to(dev)
    mouth_box = (slice(int(0.62 * 256), int(0.86 * 256)), slice(int(0.33 * 256), int(0.67 * 256)))
    ck = Path("checkpoints"); ck.mkdir(exist_ok=True); path = ck / f"{a.name}.pt"
    start = 0
    if a.resume and path.exists():
        st = torch.load(path, map_location=dev)
        model.load_state_dict(st["model"]); critic.load_state_dict(st["critic"])
        opt.load_state_dict(st["opt"]); dopt.load_state_dict(st["dopt"]); start = st["iter"]
        for _ in range(start): sched.step()
    elif a.init:
        st = torch.load(a.init, map_location=dev)
        model.load_state_dict(st["model"])
        if "critic" in st: critic.load_state_dict(st["critic"])
        print(f"initialised from {a.init}", flush=True)

    it, t0 = start, time.time()
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        pred = model(x)
        use_gan = it >= a.gan_start
        if use_gan:
            d = hinge_d(critic(x, y)[0], critic(x, pred.detach())[0])
            dopt.zero_grad(set_to_none=True); d.backward(); dopt.step()
        l1 = (torch.abs(pred - y) * W).mean()
        perc = vgg(pred, y)
        line = (torch.abs(edges(pred) - edges(y)) * W).mean()
        mouth = vgg(F.interpolate(pred[..., mouth_box[0], mouth_box[1]], size=(128, 128), mode="bilinear", align_corners=False),
                    F.interpolate(y[..., mouth_box[0], mouth_box[1]], size=(128, 128), mode="bilinear", align_corners=False))
        loss = l1 + 0.2 * perc + a.line * line + 0.3 * mouth
        if use_gan:
            adv = -critic(x, pred)[0].mean(); loss = loss + a.gan_weight * adv
        if a.temporal:
            x2, pred_w, valid = jitter(x, pred)
            temp = (torch.abs(model(x2) - pred_w) * valid).sum() / (valid.sum() * 3 + 1)
            loss = loss + a.temporal * temp
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        it += 1
        if it % 100 == 0:
            msg = f"iter {it}/{a.iters}  l1 {l1.item():.3f}  perc {perc.item():.3f}  line {line.item():.3f}  mouth {mouth.item():.3f}"
            if use_gan: msg += f"  adv {adv.item():.3f}  d {d.item():.3f}"
            if a.temporal: msg += f"  flicker {temp.item():.3f}"
            print(msg + f"  {(time.time() - t0) / (it - start):.2f}s/it", flush=True)
        if it % 2000 == 0 or it == a.iters:
            torch.save({"model": model.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                        "dopt": dopt.state_dict(), "iter": it, "arch": "face", "width": a.width}, path)
            torchvision.utils.save_image((torch.cat([x[:4], pred[:4].detach(), y[:4]], 0) + 1) / 2,
                                         ck / f"{a.name}_preview_{it:06d}.jpg", nrow=4)
        if it >= a.iters: break
