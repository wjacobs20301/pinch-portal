"""Phase 3: train FastStudent2x — 448x336 in, 896x672 out — on the full-quality, face-focused teacher data.

Each training example: the teacher's photo at half resolution as input, the teacher's 896px anime drawing as target.
Loss (faces count more everywhere — face boxes come from NAME_face.txt, mouth boxes from NAME_mouth.txt):
  * L1 + VGG perceptual, face pixels weighted 3x and mouth pixels 6x, plus close-up VGG on each face and mouth
  * line loss: edge maps (Sobel) must match, so outlines come out crisp instead of soft
  * two conditional PatchGAN critics (full and half resolution) + a dedicated face critic on face close-ups
  * anti-flicker: a slightly shifted/zoomed/brighter/noisier copy of the input must be drawn the same (warped)
Checkpoints: checkpoints/<name>.pt; previews (input / student / teacher) checkpoints/<name>_preview_*.jpg.

  .venv/bin/python train_hr.py --data data/pairs_v7c,data/faces_v7c --init checkpoints/student_v6_fast.pt
"""
import argparse, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.parametrizations import spectral_norm
from PIL import Image
from student import FastStudent2x

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="student_v8_hr")
ap.add_argument("--data", default="data/pairs_v7c,data/faces_v7c")
ap.add_argument("--init", default="", help="load body weights from a FastStudent checkpoint")
ap.add_argument("--width", type=int, default=48); ap.add_argument("--hr", type=int, default=16)
ap.add_argument("--iters", type=int, default=200000); ap.add_argument("--batch", type=int, default=6)
ap.add_argument("--crop", type=int, default=512, help="target crop (input crop is half)")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--gan-weight", type=float, default=0.1); ap.add_argument("--gan-start", type=int, default=3000)
ap.add_argument("--temporal", type=float, default=3.0); ap.add_argument("--line", type=float, default=1.0)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Pairs(Dataset):
    """Folders are sampled equally (e.g. 50% scenes, 50% face close-ups).
    Returns input (half res), target, and a 2-channel mask: face, mouth."""
    def __init__(self, roots, crop):
        self.groups = [sorted(Path(r).glob("*_in.jpg")) for r in roots.split(",")]
        self.crop = crop

    def __len__(self): return 10 ** 7

    def __getitem__(self, k):
        g = self.groups[k % len(self.groups)]
        p = g[random.randrange(len(g))]
        x = Image.open(p).convert("RGB")
        y = Image.open(str(p)[: -len("_in.jpg")] + "_out.jpg").convert("RGB").resize(x.size)
        W, H = x.size
        face = Path(str(p)[: -len("_in.jpg")] + "_face.txt")
        box = [float(v) for v in face.read_text().split()] if face.exists() else None
        mf = Path(str(p)[: -len("_in.jpg")] + "_mouth.txt")
        mbox = [float(v) for v in mf.read_text().split()] if mf.exists() else None
        s = random.uniform(0.75, 1.0)                                   # mild zoom variety
        w, h = max(self.crop, round(W * s)), max(self.crop, round(H * s))
        x, y = x.resize((w, h), Image.LANCZOS), y.resize((w, h), Image.LANCZOS)
        if box and random.random() < 0.7:                               # usually crop around the face
            cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
            i = int(min(max(cy - self.crop / 2 + random.uniform(-0.2, 0.2) * self.crop, 0), h - self.crop))
            j = int(min(max(cx - self.crop / 2 + random.uniform(-0.2, 0.2) * self.crop, 0), w - self.crop))
        else:
            i, j = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
        x, y = TF.crop(x, i, j, self.crop, self.crop), TF.crop(y, i, j, self.crop, self.crop)
        mask = torch.zeros(2, self.crop, self.crop)
        if box:
            x0, y0, x1, y1 = box[0] * w - j, box[1] * h - i, box[2] * w - j, box[3] * h - i
            pad = 0.15 * (x1 - x0)                                     # include hairline and chin
            mask[0, max(0, int(y0 - pad)):max(0, int(y1 + pad)), max(0, int(x0 - pad)):max(0, int(x1 + pad))] = 1
        if mbox:
            x0, y0, x1, y1 = mbox[0] * w - j, mbox[1] * h - i, mbox[2] * w - j, mbox[3] * h - i
            mask[1, max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = 1
        if random.random() < 0.5: x, y, mask = TF.hflip(x), TF.hflip(y), mask.flip(-1)
        x = x.resize((self.crop // 2, self.crop // 2), Image.LANCZOS)  # the app feeds 448x336 crops of the camera
        x, y = TF.to_tensor(x) * 2 - 1, TF.to_tensor(y) * 2 - 1
        x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)).clamp(-1, 1)   # webcam variety
        x = (x + torch.randn_like(x) * random.uniform(0, 0.025)).clamp(-1, 1)
        return x, y, mask


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        f = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:23].eval()
        for p in f.parameters(): p.requires_grad = False
        self.f, self.taps = f, {3, 8, 15, 22}
        self.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, a, b):
        a, b = ((a + 1) / 2 - self.m) / self.s, ((b + 1) / 2 - self.m) / self.s
        loss = 0
        for i, layer in enumerate(self.f):
            a, b = layer(a), layer(b)
            if i in self.taps: loss = loss + F.l1_loss(a, b)
        return loss


class Critic(nn.Module):
    """Conditional PatchGAN: (input upsampled to target size, drawing) → real/fake score per patch."""
    def __init__(self, w=64, cin=6):
        super().__init__()
        def blk(i, o, s): return nn.Sequential(spectral_norm(nn.Conv2d(i, o, 4, s, 1)), nn.LeakyReLU(0.2, True))
        self.blocks = nn.ModuleList([blk(cin, w, 2), blk(w, w * 2, 2), blk(w * 2, w * 4, 2), blk(w * 4, w * 8, 1)])
        self.head = spectral_norm(nn.Conv2d(w * 8, 1, 4, 1, 1))

    def forward(self, x, y):
        h, feats = torch.cat([x, y], 1), []
        for b in self.blocks: h = b(h); feats.append(h)
        return self.head(h), feats


def edges(img):
    """Sobel edge magnitude of luminance — where the line art is."""
    l = (0.299 * img[:, :1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3])
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    return torch.sqrt(F.conv2d(l, kx, padding=1) ** 2 + F.conv2d(l, kx.transpose(2, 3), padding=1) ** 2 + 1e-6)


def face_crops(img, mask, size=128, ch=0, min_px=32 * 32):
    """Crop each sample's face (ch=0) or mouth (ch=1) box from the mask and resize, for the face critic / close-up VGG.
    Skips samples without one (or where it's mostly outside the crop)."""
    out, idx = [], []
    for n in range(img.shape[0]):
        ys, xs = torch.nonzero(mask[n, ch], as_tuple=True)
        if len(ys) < min_px: continue
        y0, y1, x0, x1 = ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1
        out.append(F.interpolate(img[n:n + 1, :, y0:y1, x0:x1], size=(size, size), mode="bilinear", align_corners=False))
        idx.append(n)
    return (torch.cat(out), idx) if out else (None, [])


def jitter(x, y):
    """A slightly different 'next video frame' for the low-res input x, with the same warp at y's resolution."""
    n = x.shape[0]
    ang = (torch.rand(n, device=x.device) - 0.5) * 0.06
    sc = 1 + (torch.rand(n, device=x.device) - 0.5) * 0.06
    t = (torch.rand(n, 2, device=x.device) - 0.5) * 0.05
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
    warp = lambda img: F.grid_sample(img, F.affine_grid(theta, img.shape, align_corners=False), align_corners=False)
    x2 = warp(x) * (1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2)
    x2 = (x2 + torch.randn_like(x2) * 0.02).clamp(-1, 1)
    return x2, warp(y), warp(torch.ones_like(y[:, :1])) > 0.999


def hinge_d(real, fake): return F.relu(1 - real).mean() + F.relu(1 + fake).mean()


if __name__ == "__main__":
    ds = Pairs(a.data, a.crop)
    print("training pairs:", [len(g) for g in ds.groups], flush=True)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=6, persistent_workers=True)
    model = FastStudent2x(a.width, hr=a.hr).to(dev)
    critics = nn.ModuleList([Critic(), Critic(), Critic(48)]).to(dev)     # full res, half res, face
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.5, 0.999))
    dopt = torch.optim.Adam(critics.parameters(), lr=a.lr, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    vgg = VGG().to(dev)
    ck = Path("checkpoints"); ck.mkdir(exist_ok=True); path = ck / f"{a.name}.pt"
    start = 0
    if a.resume and path.exists():
        st = torch.load(path, map_location=dev)
        model.load_state_dict(st["model"]); critics.load_state_dict(st["critics"])
        opt.load_state_dict(st["opt"]); dopt.load_state_dict(st["dopt"]); start = st["iter"]
        for _ in range(start): sched.step()
    elif a.init:
        st = torch.load(a.init, map_location=dev)
        body = {k: v for k, v in st["model"].items() if not k.startswith("out.")}
        missing, _ = model.body.load_state_dict(body, strict=False)
        print(f"body initialised from {a.init} (missing: {missing})", flush=True)

    it, t0 = start, time.time()
    for x, y, mask in dl:
        x, y, mask = x.to(dev), y.to(dev), mask.to(dev)
        xu = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)   # condition for the critics
        pred = model(x)
        use_gan = it >= a.gan_start
        fy, fidx = face_crops(y, mask)

        if use_gan:   # ---- critics: teacher drawing = real, student drawing = fake
            p_ = pred.detach()
            d = hinge_d(critics[0](xu, y)[0], critics[0](xu, p_)[0])
            half = lambda t: F.avg_pool2d(t, 2)
            d = d + hinge_d(critics[1](half(xu), half(y))[0], critics[1](half(xu), half(p_))[0])
            if fy is not None:
                fp = face_crops(p_, mask)[0]; fx = face_crops(xu, mask)[0]
                d = d + hinge_d(critics[2](fx, fy)[0], critics[2](fx, fp)[0])
            dopt.zero_grad(set_to_none=True); d.backward(); dopt.step()

        # ---- student
        wmap = 1 + 2 * mask[:, :1] + 3 * mask[:, 1:]                         # faces count 3x, mouths 6x
        l1 = (torch.abs(pred - y) * wmap).mean()
        perc = vgg(pred, y)
        line = (torch.abs(edges(pred) - edges(y)) * wmap).mean()
        loss = l1 + 0.2 * perc + a.line * line
        fp = None
        if fy is not None:
            fp = face_crops(pred, mask)[0]
            loss = loss + 0.2 * vgg(fp, fy)                                  # extra perceptual loss on faces
        my = face_crops(y, mask, size=96, ch=1, min_px=12 * 12)[0]
        if my is not None:                                                   # and a close-up one on mouths
            loss = loss + 0.3 * vgg(face_crops(pred, mask, size=96, ch=1, min_px=12 * 12)[0], my)
        if use_gan:
            adv = -critics[0](xu, pred)[0].mean() - critics[1](F.avg_pool2d(xu, 2), F.avg_pool2d(pred, 2))[0].mean()
            if fp is not None:
                adv = adv - 2 * critics[2](face_crops(xu, mask)[0], fp)[0].mean()   # faces matter most
            loss = loss + a.gan_weight * adv
        if a.temporal:
            x2, pred_w, valid = jitter(x, pred)
            temp = (torch.abs(model(x2) - pred_w) * valid).sum() / (valid.sum() * 3 + 1)
            loss = loss + a.temporal * temp
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        it += 1

        if it % 100 == 0:
            msg = f"iter {it}/{a.iters}  l1 {l1.item():.3f}  perc {perc.item():.3f}  line {line.item():.3f}"
            if use_gan: msg += f"  adv {adv.item():.3f}  d {d.item():.3f}"
            if a.temporal: msg += f"  flicker {temp.item():.3f}"
            print(msg + f"  {(time.time() - t0) / (it - start):.2f}s/it", flush=True)
        if it % 2000 == 0 or it == a.iters:
            torch.save({"model": model.state_dict(), "critics": critics.state_dict(), "opt": opt.state_dict(),
                        "dopt": dopt.state_dict(), "iter": it, "arch": "fast2x", "width": a.width, "hr": a.hr}, path)
            grid = torch.cat([xu[:3], pred[:3].detach(), y[:3]], 0)
            torchvision.utils.save_image((grid + 1) / 2, ck / f"{a.name}_preview_{it:06d}.jpg", nrow=3)
        if it >= a.iters: break
