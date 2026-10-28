"""Shared training pieces: VGG perceptual loss, conditional PatchGAN critic, Sobel edges (line loss),
the anti-flicker jitter, and the hinge loss. Used by train_face.py (train_hr.py keeps its own copies)."""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torch.nn.utils.parametrizations import spectral_norm


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
