"""
Citizen photo analysis.

Estimates how hazy/smoky a photo is and what it shows, without a heavy deep
learning model, so it runs anywhere (including free cloud hosting):

* Haze level from the dark channel prior (He, Sun & Tang, CVPR 2009): in clear
  outdoor images most patches contain some very dark pixel; haze and smoke
  lift those dark pixels towards the colour of the atmospheric light. The
  estimated transmission map tells us how much of the scene is veiled.
* Colour cues for fire (bright orange/red), smoke (grey, low saturation),
  dust (brown-yellow cast) and clear sky (blue).
* EXIF metadata (GPS position and capture time) to geotag the report and
  check that it is recent.

The output is indicative evidence that is fused with sensor data - it is not
a PM2.5 measurement.
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
from PIL import ExifTags, Image
from scipy.ndimage import minimum_filter

CATEGORIES = ["Fire / open burning", "Smoke / open burning", "Industrial smoke / stack",
              "Dust / construction", "Haze / smog", "Clear sky", "Vehicle exhaust / congestion"]

KIND_OF_CATEGORY = {"Fire / open burning": "burning", "Smoke / open burning": "burning",
                    "Industrial smoke / stack": "industrial", "Dust / construction": "dust",
                    "Haze / smog": "haze", "Clear sky": "clear", "Vehicle exhaust / congestion": "traffic"}


def _load(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img))
    return Image.open(img)


def _to_deg(v) -> float:
    d, m, s = (float(x) for x in v)
    return d + m / 60.0 + s / 3600.0


def read_exif(im: Image.Image) -> dict:
    out = {"lat": None, "lon": None, "taken": None}
    try:
        exif = im.getexif()
    except Exception:  # noqa: BLE001
        return out
    if not exif:
        return out
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    try:
        sub = exif.get_ifd(0x8769)  # Exif IFD (DateTimeOriginal lives here)
        tags.update({ExifTags.TAGS.get(k, k): v for k, v in sub.items()})
    except Exception:  # noqa: BLE001
        pass
    dt = tags.get("DateTimeOriginal") or tags.get("DateTime")
    if dt:
        try:
            out["taken"] = datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    try:
        gps = exif.get_ifd(0x8825)
        if gps:
            g = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
            if "GPSLatitude" in g and "GPSLongitude" in g:
                lat = _to_deg(g["GPSLatitude"]) * (-1 if g.get("GPSLatitudeRef") == "S" else 1)
                lon = _to_deg(g["GPSLongitude"]) * (-1 if g.get("GPSLongitudeRef") == "W" else 1)
                out["lat"], out["lon"] = round(lat, 6), round(lon, 6)
    except Exception:  # noqa: BLE001
        pass
    return out


def dark_channel(arr: np.ndarray, patch: int = 15) -> np.ndarray:
    return minimum_filter(arr.min(axis=2), size=patch)


def _hsv(arr: np.ndarray):
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    v = mx
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    h = np.where(mx - mn < 1e-6, 0, h)
    return h, s, v


def analyze_image(img) -> dict:
    """Analyse a photo. Returns scores, a category guess and a transmission map for display."""
    im = _load(img)
    exif = read_exif(im)
    im = im.convert("RGB")
    im.thumbnail((640, 640))
    arr = np.asarray(im).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    patch = max(7, int(min(H, W) / 30) | 1)

    # --- dark channel prior
    dc = dark_channel(arr, patch)
    flat = dc.ravel()
    n_top = max(1, int(flat.size * 0.001))
    idx = np.argpartition(flat, -n_top)[-n_top:]
    A = arr.reshape(-1, 3)[idx].mean(axis=0)
    A = np.clip(A, 0.3, 1.0)
    trans = 1.0 - 0.95 * dark_channel(arr / A[None, None, :], patch)
    trans = np.clip(trans, 0.05, 1.0)
    # The prior fails on bright sky, so judge haze by the *clearest* parts of the scene
    # (dark objects): in clear air they keep high transmission, in haze even they are veiled.
    haze_index = float(np.clip((1.0 - np.percentile(trans, 80)) / 0.75, 0, 1))

    # --- colour statistics
    h, s, v = _hsv(arr)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    contrast = float(lum.std())
    fire = (h >= 12) & (h <= 50) & (s > 0.55) & (v > 0.7)
    red_light = ((h < 12) | (h > 348)) & (s > 0.6) & (v > 0.6)
    grey = (s < 0.16) & (v > 0.25) & (v < 0.85)
    sky_blue = (h > 185) & (h < 250) & (s > 0.22) & (v > 0.45)
    warm_cast = (h > 18) & (h < 55) & (s > 0.18) & (s < 0.6) & (v > 0.35)
    top = slice(0, int(H * 0.6))
    fire_frac = float(fire.mean())
    smoke_frac = float(grey[top].mean())
    blue_frac = float(sky_blue[top].mean())
    dust_frac = float(warm_cast.mean())
    lower = slice(int(H * 0.55), H)
    red_frac = float(red_light[lower].mean())
    # Plumes: grey pixels that stand out from the rest of their sky row (uniform haze does not).
    band = slice(0, max(4, int(H * 0.32)))
    vb, sb = v[band], s[band]
    row_med = np.median(vb, axis=1, keepdims=True)
    dev = np.abs(vb - row_med)
    plume_mask = (dev > 0.07) & (sb < 0.25)
    plume_score = float(np.clip(plume_mask.mean() / 0.12, 0, 1))
    darkness = float(np.median((vb / np.maximum(row_med, 1e-3))[plume_mask])) if plume_mask.any() else 1.0
    mean_sat = float(s.mean())

    # --- rule-based category scores (interpretable)
    dark_plume = 1.0 if darkness < 0.75 else 0.35
    scores = {
        "Fire / open burning": 6.0 * fire_frac / 0.03 if fire_frac > 0.004 else 0.0,
        "Smoke / open burning": 3.2 * plume_score * dark_plume + (0.8 if fire_frac > 0.004 else 0.0),
        "Industrial smoke / stack": 3.0 * plume_score * (1.3 - dark_plume) + 0.3 * haze_index,
        "Dust / construction": 1.5 * dust_frac / 0.35 + 0.8 * haze_index - 1.0 * blue_frac / 0.2,
        "Haze / smog": 1.8 * haze_index + 0.6 * (1 - mean_sat) - 1.2 * blue_frac / 0.2 - 2.0 * plume_score,
        "Clear sky": 2.0 * blue_frac / 0.2 + 1.5 * (1 - haze_index) - 1.0,
        "Vehicle exhaust / congestion": (2.4 + 0.8 * haze_index) if 0.0005 < red_frac < 0.03 and fire_frac < 0.004 else 0.0,
    }
    order = sorted(scores, key=scores.get, reverse=True)
    cat = order[0]
    z = np.array([scores[k] for k in CATEGORIES])
    p = np.exp(z - z.max())
    p = p / p.sum()
    conf = float(p[CATEGORIES.index(cat)])

    if haze_index < 0.35:
        vis = "Good (clear view)"
    elif haze_index < 0.55:
        vis = "Moderate haze"
    elif haze_index < 0.75:
        vis = "Poor - heavy haze/smoke"
    else:
        vis = "Very poor - dense smoke/smog"

    return {
        "category": cat, "category_confidence": conf,
        "category_scores": {k: float(v) for k, v in zip(CATEGORIES, p)},
        "haze_index": haze_index, "visibility": vis,
        "contrast": contrast, "fire_fraction": fire_frac, "smoke_fraction": smoke_frac,
        "blue_sky_fraction": blue_frac, "dust_cast_fraction": dust_frac, "plume_score": plume_score,
        "atmospheric_light": [float(x) for x in A],
        "transmission_map": (trans * 255).astype(np.uint8),
        "exif": exif, "size": (W, H),
    }


def report_credibility(analysis: dict, reporter_trust: float = 0.6, sensor_excess: float | None = None,
                       minutes_old: float | None = None, has_gps: bool = False) -> tuple[float, list[str]]:
    """Combine evidence into a 0-1 credibility score for a citizen report."""
    score, notes = 0.25, []
    if has_gps:
        score += 0.15
        notes.append("Photo carries GPS metadata")
    if minutes_old is not None:
        if minutes_old <= 120:
            score += 0.1
            notes.append("Captured within the last 2 hours")
        elif minutes_old > 24 * 60:
            score -= 0.15
            notes.append("Photo is more than a day old")
    if analysis["category"] != "Clear sky" and analysis["haze_index"] > 0.45:
        score += 0.15
        notes.append(f"Image shows {analysis['category'].lower()} (haze index {analysis['haze_index']:.2f})")
    if sensor_excess is not None:
        if sensor_excess > 25:
            score += 0.25
            notes.append(f"Nearby sensors confirm +{sensor_excess:.0f} µg/m³ above local background")
        elif sensor_excess < 5:
            notes.append("Nearby sensors do not (yet) show a local rise")
    score += 0.2 * (reporter_trust - 0.5)
    return float(np.clip(score, 0.02, 0.99)), notes
