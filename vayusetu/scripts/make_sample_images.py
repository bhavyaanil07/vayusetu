"""
Generate illustrative sample photos (procedurally drawn scenes) used for the
demo and for simulated citizen reports. Real photos can be uploaded in the app.

Output: data/samples/*.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "samples"
W, H = 960, 640
RNG = np.random.default_rng(11)


def sky(top, bottom):
    t = np.linspace(0, 1, H)[:, None, None]
    return (np.array(top)[None, None, :] * (1 - t) + np.array(bottom)[None, None, :] * t).repeat(W, axis=1)


def skyline(draw, base_y, color, rng, n=18, max_h=260, min_h=80, x0=0, x1=W):
    x = x0
    while x < x1:
        w = int(rng.integers(35, 90))
        h = int(rng.integers(min_h, max_h))
        draw.rectangle([x, base_y - h, x + w, base_y], fill=color)
        if rng.random() < 0.3:  # rooftop tank / antenna
            draw.rectangle([x + w // 3, base_y - h - 18, x + w // 3 + 8, base_y - h], fill=color)
        x += w + int(rng.integers(0, 12))


def depth_layers(rng, far_color, near_color):
    """Return (image, depth) with two building layers (far and near) and ground."""
    img = Image.new("RGB", (W, H), (0, 0, 0))
    depth = np.ones((H, W), np.float32) * 5.0  # sky = far
    mask_far = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask_far)
    skyline(d, int(H * 0.72), 255, rng, max_h=230, min_h=90)
    mask_near = Image.new("L", (W, H), 0)
    d2 = ImageDraw.Draw(mask_near)
    skyline(d2, int(H * 0.80), 255, rng, max_h=200, min_h=60)
    mf, mn = np.asarray(mask_far) > 0, np.asarray(mask_near) > 0
    depth[mf] = 2.0
    depth[mn] = 1.0
    depth[int(H * 0.80):] = 0.4
    return mf, mn, depth


def compose(sky_rgb, mf, mn, far_color, near_color, ground_color, depth, A, beta, rng):
    J = sky_rgb.copy()
    J[mf] = far_color
    J[mn] = near_color
    J[int(H * 0.80):] = ground_color
    # window lights / texture
    J = J + rng.normal(0, 0.015, J.shape)
    t = np.exp(-beta * depth)[..., None]
    I = J * t + np.array(A)[None, None, :] * (1 - t)
    return np.clip(I, 0, 1)


def to_img(arr) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def plume(arr, x_center, base_y, color, height, spread, rng, density=0.85, blobs=90):
    a = arr.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    acc = np.zeros((H, W), np.float32)
    for i in range(blobs):
        f = i / blobs
        y = base_y - f * height
        x = x_center + spread * f * 1.8 + rng.normal(0, 12 + 40 * f)
        r = 18 + 90 * f + rng.normal(0, 8)
        acc += np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * r ** 2))) * (1 - 0.5 * f)
    acc = np.clip(acc / acc.max() * 1.3, 0, 1) * density
    noise = np.asarray(Image.fromarray((rng.random((H // 8, W // 8)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)) / 255.0
    acc = acc * (0.75 + 0.35 * noise)
    return a * (1 - acc[..., None]) + np.array(color)[None, None, :] * acc[..., None]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    mf, mn, depth = depth_layers(rng, None, None)

    # 1. clear skyline
    s = sky([0.25, 0.52, 0.86], [0.70, 0.84, 0.96])
    img = compose(s, mf, mn, [0.30, 0.34, 0.40], [0.16, 0.18, 0.22], [0.28, 0.40, 0.22], depth, [0.85, 0.9, 0.95], 0.08, rng)
    to_img(img).filter(ImageFilter.GaussianBlur(0.6)).save(OUT / "clear_skyline.jpg", quality=90)

    # 2. hazy skyline (smog)
    s = sky([0.70, 0.70, 0.68], [0.80, 0.78, 0.72])
    img = compose(s, mf, mn, [0.30, 0.32, 0.34], [0.16, 0.17, 0.19], [0.30, 0.32, 0.24], depth, [0.80, 0.78, 0.72], 0.75, rng)
    to_img(img).filter(ImageFilter.GaussianBlur(1.0)).save(OUT / "hazy_skyline.jpg", quality=90)

    # 3. smoke plume from open burning (fire at the base)
    s = sky([0.55, 0.62, 0.72], [0.78, 0.78, 0.74])
    base = compose(s, mf, mn, [0.30, 0.32, 0.35], [0.15, 0.16, 0.18], [0.33, 0.30, 0.22], depth, [0.78, 0.76, 0.72], 0.35, rng)
    base = plume(base, 420, int(H * 0.84), [0.35, 0.34, 0.33], 520, 120, rng, density=0.9)
    im = to_img(base)
    d = ImageDraw.Draw(im)
    for _ in range(70):
        x = 420 + rng.normal(0, 45)
        y = int(H * 0.84) + rng.normal(0, 10)
        r = rng.uniform(6, 22)
        col = (255, int(rng.uniform(90, 170)), int(rng.uniform(0, 40)))
        d.ellipse([x - r, y - r * 1.6, x + r, y + r * 0.4], fill=col)
    im.filter(ImageFilter.GaussianBlur(1.2)).save(OUT / "smoke_plume.jpg", quality=90)

    # 4. industrial stack
    s = sky([0.62, 0.66, 0.72], [0.80, 0.80, 0.78])
    base = compose(s, mf, mn, [0.33, 0.34, 0.36], [0.20, 0.20, 0.22], [0.32, 0.31, 0.28], depth, [0.80, 0.79, 0.76], 0.3, rng)
    im = to_img(base)
    d = ImageDraw.Draw(im)
    for x in (600, 680):
        d.polygon([(x, int(H * 0.80)), (x + 34, int(H * 0.80)), (x + 26, int(H * 0.30)), (x + 8, int(H * 0.30))], fill=(70, 66, 64))
        d.rectangle([x + 6, int(H * 0.33), x + 28, int(H * 0.35)], fill=(170, 40, 40))
    arr = np.asarray(im).astype(np.float32) / 255
    arr = plume(arr, 617, int(H * 0.30), [0.62, 0.62, 0.62], 260, -140, rng, density=0.8, blobs=70)
    arr = plume(arr, 697, int(H * 0.30), [0.55, 0.55, 0.55], 240, -120, rng, density=0.75, blobs=60)
    to_img(arr).filter(ImageFilter.GaussianBlur(0.8)).save(OUT / "industrial_stack.jpg", quality=90)

    # 5. dust at a construction site
    s = sky([0.78, 0.70, 0.55], [0.85, 0.76, 0.58])
    base = compose(s, mf, mn, [0.45, 0.40, 0.32], [0.30, 0.26, 0.20], [0.62, 0.50, 0.34], depth, [0.84, 0.74, 0.56], 0.65, rng)
    im = to_img(base)
    d = ImageDraw.Draw(im)
    d.line([(250, int(H * 0.80)), (250, 170), (520, 170)], fill=(200, 150, 40), width=10)  # crane
    d.line([(250, 190), (330, 170)], fill=(200, 150, 40), width=6)
    d.polygon([(560, int(H * 0.86)), (700, int(H * 0.72)), (840, int(H * 0.86))], fill=(150, 120, 80))  # sand heap
    im.filter(ImageFilter.GaussianBlur(1.5)).save(OUT / "dust_site.jpg", quality=90)

    # 6. hazy traffic
    s = sky([0.72, 0.70, 0.66], [0.80, 0.77, 0.70])
    base = compose(s, mf, mn, [0.32, 0.32, 0.33], [0.18, 0.18, 0.2], [0.25, 0.25, 0.26], depth, [0.78, 0.75, 0.70], 0.7, rng)
    im = to_img(base)
    d = ImageDraw.Draw(im)
    d.polygon([(0, H), (W, H), (W * 0.58, int(H * 0.80)), (W * 0.42, int(H * 0.80))], fill=(60, 60, 64))
    for i in range(14):
        y = int(H * 0.82 + (i % 7) * 22)
        x = int(W * 0.3 + (i * 53) % 380)
        w = 60 + (i % 3) * 20
        d.rectangle([x, y, x + w, y + 30], fill=tuple(int(c) for c in rng.integers(40, 200, 3)))
        d.ellipse([x + 4, y + 18, x + 12, y + 26], fill=(255, 40, 40))
        d.ellipse([x + w - 12, y + 18, x + w - 4, y + 26], fill=(255, 40, 40))
    arr = np.asarray(im).astype(np.float32) / 255
    t = np.clip(np.linspace(0.35, 0.95, H), 0, 1)[:, None, None]
    arr = arr * t + np.array([0.76, 0.74, 0.70])[None, None, :] * (1 - t)
    to_img(arr).filter(ImageFilter.GaussianBlur(1.0)).save(OUT / "hazy_traffic.jpg", quality=90)
    print("Wrote", sorted(p.name for p in OUT.glob("*.jpg")))


if __name__ == "__main__":
    sys.exit(main())
