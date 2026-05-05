# OpenBXT — Open-source AI Deconvolution for Astronomical Images

OpenBXT is an open-source PyTorch reimplementation of the ideas behind
BlurXTerminator: an AI-based deconvolution / aberration-correction tool
specifically designed for **linear** astronomical images.

It is *not* a generative super-resolution model. The goal is to recover
detail that is actually present in the data (within the band-limit of the
optics) without inventing structure.

## Features

- U-Net with PSF-conditioned residual prediction
- Joint stellar / non-stellar handling via a learned star map
- Support for spatially-varying PSF and optical aberrations:
  - Defocus, coma, astigmatism, trefoil, spherical (Zernike-based)
  - Atmospheric seeing (Moffat/Gaussian)
  - Guiding errors (motion blur)
  - Lateral / longitudinal chromatic aberration
- Synthetic training pipeline — no real ground truth needed
- Tiled inference for arbitrarily large frames
- FITS / TIFF / PNG I/O

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Train

```bash
python train.py \
    --data_dir /path/to/sharp/linear/images \
    --out_dir runs/openbxt_v1 \
    --epochs 200 \
    --batch_size 8
```

The training set should contain *sharp* linear (un-stretched) astronomical
images. The pipeline synthesizes blurs and aberrations on-the-fly.

### Infer

```bash
python infer.py \
    --input my_image.fits \
    --output my_image_sharp.fits \
    --weights runs/openbxt_v1/best.pt \
    --sharpen_nonstellar 0.9 \
    --sharpen_stellar 0.5
```

## Project Layout

```
openbxt/
  model.py       — UNet, PSF encoder, dual-branch heads
  psf.py         — Moffat / Gaussian / Zernike PSFs
  aberrations.py — spatially-varying aberration synthesis
  dataset.py     — synthetic blur/aberration dataset
  star_detect.py — DAOFIND-style star detection
  losses.py      — flux-conserving + edge-preserving losses
  inference.py   — tiled inference engine
  utils.py       — FITS / TIFF / PNG IO, normalization
train.py
infer.py
```

## Notes on faithfulness

Deconvolution is mathematically ill-posed; many sharp images can produce the
same blurry input. OpenBXT mitigates the "hallucination" risk in three ways:

1. The training distribution **only contains realistic astronomical blurs**
   applied to real sharp astrophotos — the network never learns generic
   image priors.
2. Loss functions enforce **flux conservation** (total energy preserved
   inside aperture-like regions).
3. Output is a **bounded residual** added to the input, so under-confident
   predictions degrade gracefully toward the input rather than fabricating
   new structure.

## License

MIT.
