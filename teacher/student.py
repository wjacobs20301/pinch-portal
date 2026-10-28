"""The small, fast "student" model that learns to imitate the anime teacher in real time.

A compact U-Net: 3 downsampling stages, residual blocks at 1/8 resolution, skip connections back up.
Input/output: RGB in [-1, 1], any size divisible by 8.
"""
import torch
import torch.nn as nn


def conv(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.InstanceNorm2d(o, affine=True), nn.LeakyReLU(0.2, inplace=True))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(conv(c, c), nn.Conv2d(c, c, 3, 1, 1), nn.InstanceNorm2d(c, affine=True))

    def forward(self, x):
        return x + self.body(x)


class Up(nn.Module):
    def __init__(self, i, skip, o):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(conv(i + skip, o), conv(o, o))

    def forward(self, x, s):
        return self.conv(torch.cat([self.up(x), s], 1))


class Student(nn.Module):
    def __init__(self, w=32, blocks=4, residual=True):
        super().__init__()
        self.residual = residual   # True: output = photo + edits (v1, looked like a filter). False: repaint the whole frame.
        self.e0 = nn.Sequential(conv(3, w), conv(w, w))            # 1/1
        self.e1 = nn.Sequential(conv(w, w * 2, 2), conv(w * 2, w * 2))      # 1/2
        self.e2 = nn.Sequential(conv(w * 2, w * 4, 2), conv(w * 4, w * 4))  # 1/4
        self.e3 = conv(w * 4, w * 6, 2)                              # 1/8
        self.mid = nn.Sequential(*[Res(w * 6) for _ in range(blocks)])
        self.u2 = Up(w * 6, w * 4, w * 4)
        self.u1 = Up(w * 4, w * 2, w * 2)
        self.u0 = Up(w * 2, w, w)
        self.out = nn.Conv2d(w, 3, 3, 1, 1)

    def forward(self, x):
        s0 = self.e0(x); s1 = self.e1(s0); s2 = self.e2(s1)
        y = self.mid(self.e3(s2))
        y = self.u0(self.u1(self.u2(y, s2), s1), s0)
        y = self.out(y)
        return torch.tanh(y + x if self.residual else y)


class FastStudent(nn.Module):
    """Browser-speed variant (~20 ms at 448x336 on WebGPU in fp16, vs ~67 ms for Student(32)).
    Runs at half resolution (pixel unshuffle/shuffle keeps every pixel's information), and uses BatchNorm + ReLU,
    which fold into the convolutions at export, instead of InstanceNorm, whose reductions are slow on WebGPU."""
    def __init__(self, w=48, blocks=4):
        super().__init__()
        def c(i, o, s=1): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.inp = nn.PixelUnshuffle(2)
        self.e0 = nn.Sequential(c(12, w), c(w, w))
        self.e1 = nn.Sequential(c(w, w * 2, 2), c(w * 2, w * 2))
        self.e2 = c(w * 2, w * 3, 2)
        self.mid = nn.Sequential(*[c(w * 3, w * 3) for _ in range(blocks)])
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d1 = nn.Sequential(c(w * 3 + w * 2, w * 2), c(w * 2, w * 2))
        self.d0 = nn.Sequential(c(w * 2 + w, w), c(w, w))
        self.out = nn.Sequential(nn.Conv2d(w, 12, 3, 1, 1), nn.PixelShuffle(2))

    def forward(self, x):
        s0 = self.e0(self.inp(x)); s1 = self.e1(s0); y = self.mid(self.e2(s1))
        y = self.d1(torch.cat([self.up(y), s1], 1)); y = self.d0(torch.cat([self.up(y), s0], 1))
        return torch.tanh(self.out(y))


class FaceStudent(nn.Module):
    """Dedicated face model: an upright, aligned face crop (256x256, eyes and mouth always in the same place) in, the
    anime face out. Unlike FastStudent it works at FULL input resolution, so lips, teeth and eyes get real pixels."""
    def __init__(self, w=32, blocks=4):
        super().__init__()
        def c(i, o, s=1): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.e0 = nn.Sequential(c(3, w), c(w, w))                              # 256
        self.e1 = nn.Sequential(c(w, w * 2, 2), c(w * 2, w * 2))               # 128
        self.e2 = nn.Sequential(c(w * 2, w * 4, 2), c(w * 4, w * 4))           # 64
        self.e3 = c(w * 4, w * 6, 2)                                           # 32
        self.mid = nn.Sequential(*[c(w * 6, w * 6) for _ in range(blocks)])
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2 = nn.Sequential(c(w * 6 + w * 4, w * 4), c(w * 4, w * 4))
        self.d1 = nn.Sequential(c(w * 4 + w * 2, w * 2), c(w * 2, w * 2))
        self.d0 = nn.Sequential(c(w * 2 + w, w), c(w, w))
        self.out = nn.Conv2d(w, 3, 3, 1, 1)

    def forward(self, x):
        s0 = self.e0(x); s1 = self.e1(s0); s2 = self.e2(s1); y = self.mid(self.e3(s2))
        y = self.d2(torch.cat([self.up(y), s2], 1)); y = self.d1(torch.cat([self.up(y), s1], 1))
        y = self.d0(torch.cat([self.up(y), s0], 1))
        return torch.tanh(self.out(y))


class FastStudent2x(nn.Module):
    """FastStudent body that draws at 2x the input resolution (448x336 in → 896x672 out), so line art is drawn at
    display resolution instead of being upscaled. The body runs at half the input resolution; a light full-res stage
    (fed with the input's own fine detail) refines it, and a pixel shuffle produces the 2x output."""
    def __init__(self, w=48, blocks=4, hr=24):
        super().__init__()
        self.body = FastStudent(w, blocks)
        self.body.out = nn.Identity()                                     # reuse everything up to the last features
        def c(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.lift = nn.Sequential(nn.Conv2d(w, hr * 4, 3, 1, 1), nn.PixelShuffle(2))   # half res → input res
        self.fine = c(3, hr)                                              # detail straight from the input pixels
        self.hr = nn.Sequential(c(hr * 2, hr), c(hr, hr))
        self.out = nn.Sequential(nn.Conv2d(hr, 12, 3, 1, 1), nn.PixelShuffle(2))     # input res → 2x

    def forward(self, x):
        b = self.body
        s0 = b.e0(b.inp(x)); s1 = b.e1(s0); y = b.mid(b.e2(s1))
        y = b.d1(torch.cat([b.up(y), s1], 1)); y = b.d0(torch.cat([b.up(y), s0], 1))
        y = self.hr(torch.cat([self.lift(y), self.fine(x)], 1))
        return torch.tanh(self.out(y))


def build(st):
    """Recreate the right model from a checkpoint dict."""
    if st.get("arch") == "face": return FaceStudent(st["width"])
    if st.get("arch") == "fast2x": return FastStudent2x(st["width"], hr=st.get("hr", 24))
    if st.get("arch") == "fast": return FastStudent(st["width"])
    return Student(st.get("width", 24), residual=st.get("residual", True))


if __name__ == "__main__":
    import time
    for w in (24, 32):
        m = Student(w).eval().to("mps")
        n = sum(p.numel() for p in m.parameters())
        for (h, wd) in ((288, 512), (360, 640)):
            x = torch.randn(1, 3, h, wd, device="mps")
            with torch.no_grad():
                for _ in range(3): m(x)
                torch.mps.synchronize(); t = time.time()
                for _ in range(20): m(x)
                torch.mps.synchronize()
            print(f"width {w}: {n/1e6:.2f}M params, {wd}x{h}: {(time.time()-t)/20*1000:.1f} ms (MPS)")
