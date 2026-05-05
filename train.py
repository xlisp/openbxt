"""OpenBXT training script.

Example:
    python train.py --data_dir ./data/sharp --out_dir runs/v1 --epochs 200
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from openbxt.dataset import SyntheticAstroDataset, collate
from openbxt.losses import OpenBXTLoss
from openbxt.model import OpenBXT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--psf_size", type=int, default=33)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--base_channels", type=int, default=48)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--samples_per_image", type=int, default=8)
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = SyntheticAstroDataset(
        args.data_dir,
        crop=args.crop,
        psf_size=args.psf_size,
        n_channels=args.channels,
        samples_per_image=args.samples_per_image,
    )
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        collate_fn=collate,
    )

    model = OpenBXT(
        in_channels=args.channels,
        base_channels=args.base_channels,
        psf_size=args.psf_size,
    ).to(args.device)
    loss_fn = OpenBXTLoss().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(args.device, enabled=(args.device == "cuda"))

    start_epoch = 0
    best_val = math.inf
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", math.inf)

    for ep in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"ep{ep:03d}")
        running = {}
        for it, batch in enumerate(pbar):
            blurry = batch["blurry"].to(args.device, non_blocking=True)
            sharp = batch["sharp"].to(args.device, non_blocking=True)
            psf = batch["psf"].to(args.device, non_blocking=True)
            mask = batch["star_mask"].to(args.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(args.device, enabled=(args.device == "cuda")):
                out = model(blurry, psf=psf)
                loss, parts = loss_fn(out, sharp, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + float(v)
            if it % args.log_every == 0:
                pbar.set_postfix({k: f"{running[k] / (it + 1):.4f}" for k in running})

        sched.step()

        # Validation
        model.eval()
        val_total = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                blurry = batch["blurry"].to(args.device)
                sharp = batch["sharp"].to(args.device)
                psf = batch["psf"].to(args.device)
                mask = batch["star_mask"].to(args.device)
                out = model(blurry, psf=psf)
                loss, _ = loss_fn(out, sharp, mask)
                val_total += float(loss) * blurry.size(0)
                n += blurry.size(0)
        val_loss = val_total / max(1, n)
        print(f"[ep {ep}] val_loss={val_loss:.5f}")

        ck = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": ep,
            "args": vars(args),
            "best_val": best_val,
        }
        torch.save(ck, out / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            ck["best_val"] = best_val
            torch.save(ck, out / "best.pt")
            print(f"  ↳ new best: {best_val:.5f}")


if __name__ == "__main__":
    main()
