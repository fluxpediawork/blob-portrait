#!/usr/bin/env python3
"""blobfit: repaint any photo with soft colour blobs trained by gradient descent.

Starts from random blobs, renders them, compares to the photo, and nudges every
blob to reduce the error (Adam). Coarse-to-fine: whole image small, then whole
image medium, then random close-up crops at full size. New small blobs are added
where the error is largest.

Outputs (in --out):
  render.png        final blob painting at full size
  compare.jpg       photo | blobs, side by side
  zoom.jpg          close-up of the face area, photo vs blobs
  training.gif      noise -> picture, a frame per milestone
  blobs.npz         every blob's parameters (pos, scale, angle, colour, strength)
  stats.json        steps, blob count, PSNR, seconds, device

Usage:
  python blobfit.py photo.jpg --out result/                 # default: ~35 min on 2 CPU cores
  python blobfit.py photo.jpg --out result/ --preset quick  # ~8 min on 2 CPU cores
  python blobfit.py photo.jpg --out result/ --preset full   # best quality; use a GPU (CUDA or Apple MPS)
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

PRESETS = {               # steps, max blobs, long side of the fitted image
    "quick":   (300, 15000, 1000),
    "default": (1600, 45000, 1500),
    "full":    (6000, 100000, 1500),
}

# ---------------------------------------------------------------- setup
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_photo(path, aspect, long_side):
    """Open, honour EXIF rotation, centre-crop to the aspect (w/h), resize."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    if aspect:
        if w / h > aspect:                     # too wide: trim sides
            nw = int(h * aspect); x0 = (w - nw) // 2; im = im.crop((x0, 0, x0 + nw, h))
        else:                                  # too tall: trim top/bottom
            nh = int(w / aspect); y0 = (h - nh) // 2; im = im.crop((0, y0, w, y0 + nh))
    w, h = im.size
    s = long_side / max(w, h)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return im


def to_tensor(im, dev):
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).to(dev)


# ---------------------------------------------------------------- the blob model
class Blobs:
    """N anisotropic Gaussians. Positions are in [0,1] of image width/height;
    scales are a fraction of image width (log space)."""
    LOG_S_MIN, LOG_S_MAX, STRENGTH_MAX = math.log(2.5e-4), math.log(0.06), 3.0

    def __init__(self, capacity, start, dev, aspect_hw):
        self.dev, self.cap, self.n, self.aspect_hw = dev, capacity, start, aspect_hw
        g = torch.Generator(device="cpu").manual_seed(1234)
        r = lambda *s: torch.rand(*s, generator=g).to(dev)
        self.xy = r(capacity, 2)
        self.log_s = math.log(0.012) + r(capacity, 2) * (math.log(0.05) - math.log(0.012))
        self.theta = r(capacity) * math.pi
        self.rgb_logit = torch.logit(r(capacity, 3).clamp(0.03, 0.97))
        self.strength_logit = torch.zeros(capacity, device=dev)
        self.params = [self.xy, self.log_s, self.theta, self.rgb_logit, self.strength_logit]
        self.lr = [0.004, 0.02, 0.02, 0.03, 0.03]
        for p in self.params:
            p.requires_grad_(True)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = torch.zeros(capacity, device=dev)       # per-blob Adam step count

    # which blobs can touch a window (x0, y0, w, h) in pixels at width W
    @torch.no_grad()
    def visible(self, W, H, win):
        n = self.n
        if win is None:
            return torch.arange(n, device=self.dev)
        x0, y0, w, h = win
        cx, cy = self.xy[:n, 0] * W, self.xy[:n, 1] * H
        reach = 3.0 * torch.exp(self.log_s[:n]).max(1).values * W
        keep = (cx > x0 - reach) & (cx < x0 + w + reach) & (cy > y0 - reach) & (cy < y0 + h + reach)
        return torch.nonzero(keep).squeeze(1)

    def render(self, idx, W, H, win, bg):
        """Differentiable splat of blobs `idx` into window `win` of a W x H canvas.
        Each blob only writes into a square patch sized to its footprint."""
        x0, y0, w, h = win if win is not None else (0, 0, W, H)
        cx = self.xy[idx, 0] * W - x0
        cy = self.xy[idx, 1] * H - y0
        s = torch.exp(self.log_s[idx]) * W
        cos, sin = torch.cos(self.theta[idx]), torch.sin(self.theta[idx])
        isx, isy = 1 / s[:, 0] ** 2, 1 / s[:, 1] ** 2
        qa = cos * cos * isx + sin * sin * isy
        qb = cos * sin * (isx - isy)
        qc = sin * sin * isx + cos * cos * isy
        amp = self.STRENGTH_MAX * torch.sigmoid(self.strength_logit[idx])
        rgb = torch.sigmoid(self.rgb_logit[idx])
        size = s.max(1).values.detach()

        accum = torch.zeros(h * w, 4, device=self.dev)
        lo = 0.0
        for hi, patch in ((2.0, 12), (5.5, 34), (16.0, 98), (48.0, 290), (1e9, 600)):
            band = torch.nonzero((size > lo) & (size <= hi)).squeeze(1); lo = hi
            if band.numel() == 0:
                continue
            off = torch.arange(patch, device=self.dev) - patch // 2
            oy, ox = torch.meshgrid(off, off, indexing="ij")
            ox, oy = ox.reshape(1, -1), oy.reshape(1, -1)
            per_chunk = max(1, 3_500_000 // (patch * patch))
            for k in range(0, band.numel(), per_chunk):
                j = band[k:k + per_chunk]
                px = torch.floor(cx[j]).detach().long().unsqueeze(1) + ox
                py = torch.floor(cy[j]).detach().long().unsqueeze(1) + oy
                dx = px.float() + 0.5 - cx[j].unsqueeze(1)
                dy = py.float() + 0.5 - cy[j].unsqueeze(1)
                q = qa[j, None] * dx * dx + 2 * qb[j, None] * dx * dy + qc[j, None] * dy * dy
                inside = ((px >= 0) & (px < w) & (py >= 0) & (py < h)).float()
                wgt = amp[j, None] * torch.exp(-0.5 * q) * inside
                flat = (py.clamp(0, h - 1) * w + px.clamp(0, w - 1)).reshape(-1)
                vals = torch.cat([wgt.unsqueeze(2) * rgb[j].unsqueeze(1), wgt.unsqueeze(2)], 2).reshape(-1, 4)
                accum = accum.index_add(0, flat, vals)
        weight = accum[:, 3:]
        cover = 1 - torch.exp(-weight)                  # how opaque the paint is here
        colour = accum[:, :3] / (weight + 1e-5)         # weighted mean colour of blobs here
        return (bg * (1 - cover) + colour * cover).reshape(h, w, 3)

    @torch.no_grad()
    def adam_step(self, idx, scale_lr, W):
        """Adam, but only for blobs that were in view this step."""
        b1, b2 = 0.9, 0.99
        self.t[idx] += 1
        c1, c2 = 1 - b1 ** self.t[idx], 1 - b2 ** self.t[idx]
        for i, (p, m, v, base) in enumerate(zip(self.params, self.m, self.v, self.lr)):
            g = p.grad[idx]
            lr = base * scale_lr * (160.0 / W if i == 0 else 1.0)   # positions move less at high res
            m[idx] = b1 * m[idx] + (1 - b1) * g
            v[idx] = b2 * v[idx] + (1 - b2) * g * g
            shape = (-1,) + (1,) * (g.dim() - 1)
            p[idx] -= lr * (m[idx] / c1.view(shape)) / ((v[idx] / c2.view(shape)).sqrt() + 1e-8)
            p.grad = None
        self.log_s.clamp_(self.LOG_S_MIN, self.LOG_S_MAX)
        self.xy.clamp_(-0.2, 1.2)

    @torch.no_grad()
    def add_where_wrong(self, k, painted, target, W, H, win):
        """Spawn k tiny blobs at pixels sampled by error, coloured like the current paint."""
        k = min(k, self.cap - self.n)
        if k <= 0:
            return
        x0, y0, w, h = win if win is not None else (0, 0, W, H)
        err = (painted - target).abs().mean(2).reshape(-1) ** 1.5 + 1e-6
        pick = torch.multinomial(err.cpu(), k, replacement=True).to(self.dev)
        a, b = self.n, self.n + k
        self.xy[a:b, 0] = (x0 + (pick % w).float() + torch.rand(k, device=self.dev)) / W
        self.xy[a:b, 1] = (y0 + (pick // w).float() + torch.rand(k, device=self.dev)) / H
        self.log_s[a:b] = math.log(1.2 / W) + 0.15 * torch.randn(k, 2, device=self.dev)
        self.theta[a:b] = torch.rand(k, device=self.dev) * math.pi
        self.rgb_logit[a:b] = torch.logit(painted.reshape(-1, 3)[pick].clamp(0.02, 0.98))
        self.strength_logit[a:b] = -1.4
        self.n = b


# ---------------------------------------------------------------- loss
def ssim(x, y):
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    g = torch.exp(-(torch.arange(11, device=x.device) - 5.0) ** 2 / 4.5); g = g / g.sum()
    k = (g[:, None] * g[None]).expand(3, 1, 11, 11).contiguous()
    blur = lambda t: F.conv2d(t, k, padding=5, groups=3)
    mx, my = blur(x), blur(y)
    vx, vy, vxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    return (((2 * mx * my + 1e-4) * (2 * vxy + 9e-4)) / ((mx * mx + my * my + 1e-4) * (vx + vy + 9e-4))).mean()


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default="blobfit-out")
    ap.add_argument("--preset", choices=PRESETS, default="default")
    ap.add_argument("--steps", type=int); ap.add_argument("--blobs", type=int)
    ap.add_argument("--aspect", default="4:5", help="crop aspect w:h, or 'none' to keep the photo's own")
    ap.add_argument("--long-side", type=int, help="pixels on the long side of the fitted image")
    ap.add_argument("--focus", default="0.5,0.4,0.45,0.5",
                    help="cx,cy,w,h (0-1) of the area that gets most close-up crops: usually the face")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    steps, nmax, long_side = PRESETS[a.preset]
    steps, nmax, long_side = a.steps or steps, a.blobs or nmax, a.long_side or long_side
    aspect = None if a.aspect == "none" else (lambda p: float(p[0]) / float(p[1]))(a.aspect.split(":"))
    fcx, fcy, fw, fh = (float(v) for v in a.focus.split(","))
    dev = pick_device(a.device)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    photo = load_photo(a.image, aspect, long_side)
    FW, FH = photo.size
    widths = sorted({min(160, FW), min(320, FW), min(FW // 2 * 2, max(640, FW // 2)), FW})
    def size_at(W): return W, round(W * FH / FW)
    target = {W: to_tensor(photo if W == FW else photo.resize(size_at(W), Image.LANCZOS), dev) for W in widths}
    t_small = target[widths[0]]
    border = torch.cat([t_small[0], t_small[-1], t_small[:, 0], t_small[:, -1]])
    bg = border.median(0).values                         # backdrop = the photo's typical edge colour

    # schedule: share of steps per phase, and crop size for the crop phases
    A, B, C = int(steps * 0.07), int(steps * 0.17), int(steps * 0.36)
    def phase(step):
        if step < A: return widths[0], None
        if step < B: return widths[1], None
        if step < C: return widths[-2], min(288, widths[-2])
        return FW, min(352, FW)
    def target_count(step):
        pts = [(0, 1500), (A, 3500), (B, 9000), (C, int(nmax * 0.3)), (int(steps * 0.86), nmax), (steps, nmax)]
        for (s0, n0), (s1, n1) in zip(pts, pts[1:]):
            if step <= s1: return int(n0 + (n1 - n0) * (step - s0) / max(1, s1 - s0))
        return nmax
    def window(W, cs):
        H = size_at(W)[1]
        if rng.random() < 0.62:
            cx = rng.uniform(fcx - fw / 2, fcx + fw / 2); cy = rng.uniform(fcy - fh / 2, fcy + fh / 2)
        else:
            cx, cy = rng.uniform(0, 1), rng.uniform(0, 1)
        return (int(np.clip(cx * W - cs / 2, 0, W - cs)), int(np.clip(cy * H - cs / 2, 0, H - cs)), cs, cs)

    blobs = Blobs(nmax, 1500, dev, FH / FW)
    gif_at = sorted({0, 1, 5, 15, 40, 100, 250, A, B, C, int(steps * .6), int(steps * .8), steps})
    frames = []
    def keep_frame(step):
        with torch.no_grad():
            W, H = size_at(min(480, FW))
            im = blobs.render(torch.arange(blobs.n, device=dev), W, H, None, bg)
        fr = Image.fromarray((im.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        ImageDraw.Draw(fr).text((10, 10), f"STEP {step}   {blobs.n:,} blobs", fill="white")
        frames.append(fr)

    keep_frame(0); t0 = time.time()
    for step in range(1, steps + 1):
        W, cs = phase(step); H = size_at(W)[1]
        win = window(W, cs) if cs else None
        idx = blobs.visible(W, H, win)
        img = blobs.render(idx, W, H, win, bg)
        x0, y0, w, h = win if win else (0, 0, W, H)
        tg = target[W][y0:y0 + h, x0:x0 + w]
        loss = (img - tg).abs().mean()
        if W >= 320:
            loss = 0.8 * loss + 0.2 * (1 - ssim(img, tg))
        loss.backward()
        f = step / steps
        blobs.adam_step(idx, 1.0 if f < 0.5 else 0.2 ** ((f - 0.5) / 0.5), W)
        if step >= 80 and step % (40 if cs is None else 10) == 0:
            blobs.add_where_wrong(target_count(step) - blobs.n, img.detach(), tg, W, H, win)
        if step in gif_at: keep_frame(step)
        if step % 100 == 0 or step == steps:
            print(f"step {step:5d}/{steps}  {W}px  blobs {blobs.n:6d}  loss {loss.item():.4f}  {time.time() - t0:6.0f}s", flush=True)
    secs = time.time() - t0

    # ------------------------------------------------------------ outputs
    with torch.no_grad():
        final = blobs.render(torch.arange(blobs.n, device=dev), FW, FH, None, bg).clamp(0, 1)
        psnr = -10 * math.log10(((final - target[FW]) ** 2).mean().item())
    render = Image.fromarray((final.cpu().numpy() * 255).astype(np.uint8))
    render.save(os.path.join(a.out, "render.png"))

    cw = min(900, FW); ch = round(cw * FH / FW)
    comp = Image.new("RGB", (cw * 2 + 30, ch + 60), "black"); d = ImageDraw.Draw(comp)
    comp.paste(photo.resize((cw, ch), Image.LANCZOS), (10, 50)); comp.paste(render.resize((cw, ch), Image.LANCZOS), (cw + 20, 50))
    d.text((14, 18), "REAL PHOTO", fill="white")
    d.text((cw + 24, 18), f"CODE: {blobs.n:,} blobs, {steps:,} gradient-descent steps", fill="white")
    comp.save(os.path.join(a.out, "compare.jpg"), quality=92)

    zw, zh = int(FW * fw * 0.6), int(FH * fh * 0.25)
    zx, zy = int(FW * fcx - zw / 2), int(FH * (fcy + fh * 0.04) - zh / 2)
    box = (max(0, zx), max(0, zy), min(FW, zx + zw), min(FH, zy + zh))
    pw = 600; ph = round(pw * (box[3] - box[1]) / (box[2] - box[0]))
    zoom = Image.new("RGB", (pw * 2 + 30, ph + 50), "black"); d = ImageDraw.Draw(zoom)
    zoom.paste(photo.crop(box).resize((pw, ph), Image.BICUBIC), (10, 40)); zoom.paste(render.crop(box).resize((pw, ph), Image.BICUBIC), (pw + 20, 40))
    d.text((14, 14), "REAL PHOTO, zoom", fill="white"); d.text((pw + 24, 14), "CODE, zoom", fill="white")
    zoom.save(os.path.join(a.out, "zoom.jpg"), quality=92)

    frames[0].save(os.path.join(a.out, "training.gif"), save_all=True,
                   append_images=frames[1:] + [frames[-1]] * 3, duration=550, loop=0)
    n = blobs.n
    np.savez_compressed(os.path.join(a.out, "blobs.npz"), xy=blobs.xy[:n].detach().cpu().numpy(),
                        scale=torch.exp(blobs.log_s[:n]).detach().cpu().numpy(), angle=blobs.theta[:n].detach().cpu().numpy(),
                        rgb=torch.sigmoid(blobs.rgb_logit[:n]).detach().cpu().numpy(),
                        strength=(Blobs.STRENGTH_MAX * torch.sigmoid(blobs.strength_logit[:n])).detach().cpu().numpy(),
                        size=np.array([FW, FH]), background=bg.cpu().numpy())
    stats = {"steps": steps, "blobs": n, "psnr_db": round(psnr, 2), "seconds": round(secs), "device": str(dev),
             "size": [FW, FH], "preset": a.preset}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=2)
    print("done", json.dumps(stats))


if __name__ == "__main__":
    main()
