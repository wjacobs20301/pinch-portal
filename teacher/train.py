"""Train the Student to imitate the teacher's anime redraws (data/pairs/*_in.jpg → *_out.jpg).

Loss = L1 (colors/shapes) + VGG perceptual (lines and texture) [+ adversarial critic with --gan].
L1/VGG alone reward a safe, blurry average of the teacher (v1 looked like a soft light filter); the critic
(a conditional PatchGAN) learns to tell real teacher anime from student output, pushing the student toward
crisp lines and flat colors.
Checkpoints go to checkpoints/<name>.pt; preview grids (input / student / teacher) to checkpoints/<name>_preview_*.jpg.

  v1:  .venv/bin/python train.py --name student --width 24
  v2:  .venv/bin/python train.py --name student_v2 --width 32 --no-residual --gan --iters 40000 --batch 16
  v3:  see run_round3.sh (stronger teacher, 50% face close-ups, long run initialised from v2)
"""
import argparse, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.parametrizations import spectral_norm
from PIL import Image
import torchvision.transforms.functional as TF
from student import Student, FastStudent, build

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="student")
ap.add_argument("--iters", type=int, default=30000)
ap.add_argument("--width", type=int, default=24)
ap.add_argument("--no-residual", action="store_true", help="repaint the whole frame instead of editing the photo")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--gan", action="store_true", help="add the adversarial critic")
ap.add_argument("--gan-weight", type=float, default=0.05)
ap.add_argument("--gan-start", type=int, default=3000, help="iterations of plain L1/VGG warm-up before the critic kicks in")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--data", default="data/pairs", help="comma-separated pair folders; each folder is sampled equally often")
ap.add_argument("--init", default="", help="start from another checkpoint's weights (e.g. checkpoints/student_v2.pt)")
ap.add_argument("--arch", default="unet", choices=["unet", "fast"], help="fast = browser-speed FastStudent")
ap.add_argument("--distill", default="", help="also match this finished checkpoint's output (frozen), e.g. the smooth v4")
ap.add_argument("--distill-weight", type=float, default=1.0)
ap.add_argument("--temporal", type=float, default=0.0,
                help="weight of the anti-flicker loss: the same frame, slightly shifted/noisier/brighter, must be drawn the same")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Pairs(Dataset):
    def __init__(self, roots="data/pairs", crop=256):
        self.groups = [sorted(Path(r).glob("*_in.jpg")) for r in roots.split(",")]
        self.ins = [p for g in self.groups for p in g]
        self.crop = crop

    def __len__(self):
        return len(self.ins) * 1000          # sample with replacement; the iteration count controls length

    def __getitem__(self, k):
        g = self.groups[k % len(self.groups)]            # each folder equally often (e.g. 50% face close-ups)
        p = g[random.randrange(len(g))]
        x = Image.open(p).convert("RGB"); y = Image.open(str(p).replace("_in.jpg", "_out.jpg")).convert("RGB")
        # random scale so the model sees faces at many sizes, then random crop
        s = random.uniform(0.6, 1.0)
        size = (max(self.crop, int(x.height * s)), max(self.crop, int(x.width * s)))
        x, y = TF.resize(x, size, antialias=True), TF.resize(y, size, antialias=True)
        i, j = random.randint(0, size[0] - self.crop), random.randint(0, size[1] - self.crop)
        x, y = TF.crop(x, i, j, self.crop, self.crop), TF.crop(y, i, j, self.crop, self.crop)
        if random.random() < 0.5: x, y = TF.hflip(x), TF.hflip(y)
        x, y = TF.to_tensor(x) * 2 - 1, TF.to_tensor(y) * 2 - 1
        # webcam robustness: brightness/contrast jitter + sensor noise on the input only
        x = (x * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)).clamp(-1, 1)
        x = (x + torch.randn_like(x) * random.uniform(0, 0.03)).clamp(-1, 1)
        return x, y


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:23].eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg, self.layers = vgg, {3, 8, 15, 22}
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, a, b):
        a = ((a + 1) / 2 - self.mean) / self.std; b = ((b + 1) / 2 - self.mean) / self.std
        loss = 0
        for i, layer in enumerate(self.vgg):
            a, b = layer(a), layer(b)
            if i in self.layers: loss = loss + F.l1_loss(a, b)
        return loss


class Critic(nn.Module):
    """Conditional PatchGAN: sees (photo, anime) and scores each ~70px patch as teacher-real or student-fake."""
    def __init__(self, w=64):
        super().__init__()
        def block(i, o, s): return nn.Sequential(spectral_norm(nn.Conv2d(i, o, 4, s, 1)), nn.LeakyReLU(0.2, True))
        self.blocks = nn.ModuleList([block(6, w, 2), block(w, w * 2, 2), block(w * 2, w * 4, 2), block(w * 4, w * 8, 1)])
        self.head = spectral_norm(nn.Conv2d(w * 8, 1, 4, 1, 1))

    def forward(self, x, y):
        h, feats = torch.cat([x, y], 1), []
        for b in self.blocks:
            h = b(h); feats.append(h)
        return self.head(h), feats


def jitter(x, y):
    """A slightly different 'next video frame': tiny shift/zoom/rotation, plus brightness change and sensor noise
    on the input only. Returns the jittered input and the same geometric warp applied to y (with a validity mask)."""
    n = x.shape[0]
    ang = (torch.rand(n, device=x.device) - 0.5) * 0.06                 # about ±1.7 degrees
    sc = 1 + (torch.rand(n, device=x.device) - 0.5) * 0.06              # about ±3% zoom
    t = (torch.rand(n, 2, device=x.device) - 0.5) * 0.05                # about ±1.25% shift
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    warp = lambda img: F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = warp(torch.ones_like(x[:, :1])) > 0.999
    x2 = warp(x) * (1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2)
    x2 = (x2 + torch.randn_like(x2) * 0.02).clamp(-1, 1)
    return x2, warp(y), mask


def save(path, model, opt, it, critic=None, dopt=None):
    st = {"model": model.state_dict(), "opt": opt.state_dict(), "iter": it, "width": a.width, "residual": not a.no_residual,
          "arch": a.arch}
    if critic is not None: st |= {"critic": critic.state_dict(), "dopt": dopt.state_dict()}
    torch.save(st, path)


if __name__ == "__main__":
    ds = Pairs(a.data, crop=a.crop)
    print(f"{len(ds.ins)} training pairs ({', '.join(str(len(g)) for g in ds.groups)})", flush=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=6, persistent_workers=True)
    model = (FastStudent(a.width) if a.arch == "fast" else Student(a.width, residual=not a.no_residual)).to(dev)
    mentor = None
    if a.distill:   # a finished model whose (smooth, anime) output this one also learns to reproduce
        st = torch.load(a.distill, map_location=dev)
        mentor = build(st).to(dev).eval(); mentor.load_state_dict(st["model"])
        for p_ in mentor.parameters(): p_.requires_grad = False
        print(f"distilling from {a.distill}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    vgg = VGGLoss().to(dev)
    critic = Critic().to(dev) if a.gan else None
    dopt = torch.optim.Adam(critic.parameters(), lr=a.lr, betas=(0.5, 0.999)) if a.gan else None
    ck = Path("checkpoints"); ck.mkdir(exist_ok=True)
    path = ck / f"{a.name}.pt"
    start = 0
    if a.init and not (a.resume and path.exists()):
        st = torch.load(a.init, map_location=dev)
        model.load_state_dict(st["model"])
        if critic is not None and "critic" in st: critic.load_state_dict(st["critic"])
        print(f"initialised from {a.init}", flush=True)
    if a.resume and path.exists():
        st = torch.load(path, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["iter"]
        if critic is not None and "critic" in st: critic.load_state_dict(st["critic"]); dopt.load_state_dict(st["dopt"])
        for _ in range(start): sched.step()

    it, t0 = start, time.time()
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        use_gan = critic is not None and it >= a.gan_start
        pred = model(x)

        # critic step: teacher anime = real, student output = fake (hinge loss)
        if use_gan:
            real, _ = critic(x, y); fake, _ = critic(x, pred.detach())
            d_loss = F.relu(1 - real).mean() + F.relu(1 + fake).mean()
            dopt.zero_grad(set_to_none=True); d_loss.backward(); dopt.step()

        # student step
        l1 = F.l1_loss(pred, y); perc = vgg(pred, y)
        loss = l1 + 0.2 * perc
        if mentor is not None:
            with torch.no_grad(): m_out = mentor(x)
            dist = F.l1_loss(pred, m_out) + 0.2 * vgg(pred, m_out)
            loss = loss + a.distill_weight * dist
        if use_gan:
            fake, f_feats = critic(x, pred)
            with torch.no_grad(): _, r_feats = critic(x, y)
            adv = -fake.mean()
            fm = sum(F.l1_loss(f, r) for f, r in zip(f_feats, r_feats)) / len(f_feats)   # feature matching: stabilizes
            loss = loss + a.gan_weight * adv + 0.5 * a.gan_weight * 10 * fm
        if a.temporal:
            # anti-flicker: draw a jittered copy of the frame; it must match this frame's drawing, warped the same way
            x2, pred_w, mask = jitter(x, pred)
            pred2 = model(x2)
            temp = (torch.abs(pred2 - pred_w) * mask).sum() / (mask.sum() * 3 + 1)
            loss = loss + a.temporal * temp
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        it += 1

        if it % 100 == 0:
            msg = f"iter {it}/{a.iters}  l1 {l1.item():.3f}  perc {perc.item():.3f}"
            if use_gan: msg += f"  adv {adv.item():.3f}  d {d_loss.item():.3f}"
            if a.temporal: msg += f"  flicker {temp.item():.3f}"
            if mentor is not None: msg += f"  distill {dist.item():.3f}"
            print(msg + f"  {(time.time()-t0)/(it-start):.2f}s/it", flush=True)
        if it % 1000 == 0 or it == a.iters:
            save(path, model, opt, it, critic, dopt)
            grid = torch.cat([x[:4], pred[:4].detach(), y[:4]], 0)
            torchvision.utils.save_image((grid + 1) / 2, ck / f"{a.name}_preview_{it:06d}.jpg", nrow=4)
        if it >= a.iters: break
