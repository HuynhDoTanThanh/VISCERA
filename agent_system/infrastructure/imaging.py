"""Image cropping for the region/multi-scale stage. Given a normalized bbox from Stage B, crop the
focal area (with margin), upscale it, and cache it next to the raw store so the chunk prompts can
see the suspicious region at higher effective resolution.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

from PIL import Image


def crop_region(image_path: str, bbox, out_dir: Path, size: int = 512,
                margin: float = 0.15) -> str | None:
    """bbox = [x0, y0, x1, y1] normalized to [0,1]. Returns the crop path, or None if invalid."""
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        return None
    # reject degenerate / whole-frame boxes (no zoom benefit)
    if (x1 - x0) > 0.95 and (y1 - y0) > 0.95:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    W, H = img.size
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    box = (int(max(0, x0 - mx) * W), int(max(0, y0 - my) * H),
           int(min(1, x1 + mx) * W), int(min(1, y1 + my) * H))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return None
    crop = img.crop(box)
    scale = size / max(crop.size)
    if scale > 1:
        crop = crop.resize((int(crop.size[0] * scale), int(crop.size[1] * scale)), Image.LANCZOS)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{Path(image_path).name}|{box}".encode()).hexdigest()[:16]
    out = out_dir / f"{key}.png"
    if not out.exists():
        crop.save(out)
    return str(out)
