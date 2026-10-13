from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_degradation(image: Image.Image, condition: str) -> Image.Image:
    condition = condition.lower()
    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    if condition == "noise":
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.normal(0.0, 0.08, arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if condition == "low_resolution":
        small = image.resize((28, 28), Image.BICUBIC)
        return small.resize((112, 112), Image.BICUBIC)
    if condition == "brightness":
        return ImageEnhance.Brightness(image).enhance(0.50)
    if condition == "occlusion":
        out = image.copy()
        w, h = out.size
        x0 = int(w * 0.25)
        y0 = int(h * 0.35)
        patch = Image.new("RGB", (int(w * 0.50), int(h * 0.30)), (0, 0, 0))
        out.paste(patch, (x0, y0))
        return out
    return image
