"""I/O and normalization helpers.

Supports FITS (astropy), TIFF (tifffile) and PNG/JPG (PIL).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as float32 array. Returns (C, H, W) with C in {1, 3}."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".fits", ".fit", ".fts"):
        from astropy.io import fits  # type: ignore
        with fits.open(p) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    arr = np.asarray(hdu.data, dtype=np.float32)
                    break
            else:
                raise IOError(f"No image data in {p}")
        if arr.ndim == 2:
            arr = arr[None]
        elif arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] in (1, 3):
            arr = np.moveaxis(arr, -1, 0)
        return arr.astype(np.float32)
    if suf in (".tif", ".tiff"):
        import tifffile  # type: ignore
        arr = tifffile.imread(str(p))
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            arr = np.moveaxis(arr[..., :3], -1, 0)
        # Heuristic: if values look like 16-bit, normalize
        if arr.max() > 2.0:
            arr = arr / np.iinfo(np.uint16).max
        return arr
    # PIL fallback
    from PIL import Image
    img = Image.open(p)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None]
    else:
        arr = np.moveaxis(arr[..., :3], -1, 0)
    return arr


def save_image(path: str | Path, arr: np.ndarray, header=None) -> None:
    """Save (C, H, W) float array. Format inferred from extension."""
    p = Path(path)
    suf = p.suffix.lower()
    arr = np.asarray(arr, dtype=np.float32)
    if suf in (".fits", ".fit", ".fts"):
        from astropy.io import fits  # type: ignore
        out = arr if arr.shape[0] > 1 else arr[0]
        hdu = fits.PrimaryHDU(out, header=header)
        hdu.writeto(p, overwrite=True)
        return
    if suf in (".tif", ".tiff"):
        import tifffile  # type: ignore
        if arr.shape[0] in (1, 3):
            out = np.moveaxis(arr, 0, -1)
            if arr.shape[0] == 1:
                out = out[..., 0]
        else:
            out = arr
        tifffile.imwrite(str(p), out.astype(np.float32))
        return
    from PIL import Image
    a = arr.clip(0, 1) * 255.0
    if a.shape[0] == 1:
        img = Image.fromarray(a[0].astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(np.moveaxis(a, 0, -1).astype(np.uint8), mode="RGB")
    img.save(p)


def auto_stretch(arr: np.ndarray, bg_pct: float = 25.0, sat_pct: float = 99.7) -> tuple[np.ndarray, float, float]:
    """Robust [0, 1] normalization for linear astro images.

    Returns the normalized array and (lo, hi) used so it can be inverted.
    """
    lo = float(np.percentile(arr, bg_pct))
    hi = float(np.percentile(arr, sat_pct))
    if hi <= lo:
        hi = lo + 1.0
    out = (arr - lo) / (hi - lo)
    return out, lo, hi


def to_tensor(arr: np.ndarray, device=None) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).to(device)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()
