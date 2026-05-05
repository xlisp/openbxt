"""OpenBXT inference CLI.

Example:
    python infer.py --input M31.fits --output M31_sharp.fits \\
        --weights runs/v1/best.pt --sharpen_nonstellar 0.9 --sharpen_stellar 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from openbxt.inference import deconvolve
from openbxt.model import OpenBXT
from openbxt.utils import load_image, save_image, auto_stretch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--sharpen_nonstellar", type=float, default=0.9)
    p.add_argument("--sharpen_stellar", type=float, default=0.5)
    p.add_argument("--no_auto_psf", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_psf", type=str, default=None)
    p.add_argument("--save_starmap", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    ck = torch.load(args.weights, map_location=args.device)
    margs = ck.get("args", {})
    model = OpenBXT(
        in_channels=margs.get("channels", 3),
        base_channels=margs.get("base_channels", 48),
        psf_size=margs.get("psf_size", 33),
    ).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()

    arr = load_image(args.input)  # (C, H, W)
    norm, lo, hi = auto_stretch(arr)
    norm = np.clip(norm, 0.0, 1.5)

    if norm.shape[0] == 1 and model.in_channels == 3:
        norm = np.repeat(norm, 3, axis=0)
    elif norm.shape[0] >= 3 and model.in_channels == 1:
        norm = norm.mean(axis=0, keepdims=True)
    else:
        norm = norm[: model.in_channels]

    img = torch.from_numpy(norm).float().to(args.device)

    out = deconvolve(
        model,
        img,
        tile=args.tile,
        overlap=args.overlap,
        sharpen_nonstellar=args.sharpen_nonstellar,
        sharpen_stellar=args.sharpen_stellar,
        auto_psf=not args.no_auto_psf,
    )

    sharp_np = out["sharp"].cpu().numpy()
    # Denormalize back to original linear scale
    sharp_np = sharp_np * (hi - lo) + lo

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(out_path, sharp_np)
    print(f"Saved {out_path}")

    if args.save_psf:
        save_image(args.save_psf, out["psf_used"][None].cpu().numpy())
    if args.save_starmap:
        save_image(args.save_starmap, out["star_map"].cpu().numpy())


if __name__ == "__main__":
    main()
