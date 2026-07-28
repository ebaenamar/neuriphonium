"""Train EEG -> MusicCoCa cross-modal alignment on NMED-T.

Pipeline:
  1. Precompute frozen MusicCoCa audio embeddings per song (cached .npz).
  2. Load paired EEG windows (NMED-T, resampled to 200 Hz for LaBraM).
  3. Train LaBraM alignment head (optionally last backbone blocks) with a
     CLIP-style contrastive loss against the frozen MusicCoCa targets.
  4. Evaluate with leave-songs-out retrieval (Recall@K, EEG -> audio).

Usage:
  venv/bin/python train_align.py --data-root /path/to/nmed_t --epochs 30
  venv/bin/python train_align.py --data-root /path/to/nmed_t --eval-songs silent_shout oino
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_music_align.dataset import NMEDTAlignDataset, egi_chs_info
from eeg_music_align.model import CLIPLoss, EEGMusicAligner, retrieval_recall
from eeg_music_align.musiccoca_embed import MusicCoCaSongEmbedder

OUTPUT_DIR = os.path.join("output", "align")


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_eval(model, loader, device):
    model.eval()
    zs_eeg, zs_audio = [], []
    with torch.no_grad():
        for eeg, target, _ in loader:
            z = model(eeg.to(device))
            zs_eeg.append(z.cpu())
            zs_audio.append(target)
    model.train()
    return retrieval_recall(torch.cat(zs_eeg), torch.cat(zs_audio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="NMED-T root with music/ and data_processed/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--unfreeze-blocks", type=int, default=0,
                    help="Fine-tune last N LaBraM blocks (0 = frozen backbone)")
    ap.add_argument("--hop-seconds", type=float, default=10.0)
    ap.add_argument("--window-seconds", type=float, default=10.0)
    ap.add_argument("--eval-songs", nargs="*", default=["silent_shout", "oino"],
                    help="Held-out songs for retrieval eval")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=622)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[1/4] Precomputing MusicCoCa embeddings (device-independent, TFLite CPU)...")
    embedder = MusicCoCaSongEmbedder(os.path.join(OUTPUT_DIR, "musiccoca_cache.npz"))

    print(f"[2/4] Loading NMED-T pairs...")
    from eeg_music_align.dataset import NMED_T_SONGS
    all_songs = list(NMED_T_SONGS)
    eval_songs = [s for s in args.eval_songs if s in all_songs]
    fit_songs = [s for s in all_songs if s not in eval_songs]

    train_ds = NMEDTAlignDataset(args.data_root, embedder,
                                 window_seconds=args.window_seconds,
                                 hop_seconds=args.hop_seconds, songs=fit_songs)
    eval_ds = NMEDTAlignDataset(args.data_root, embedder,
                                window_seconds=args.window_seconds,
                                hop_seconds=args.hop_seconds, songs=eval_songs)
    embedder.save()
    n_train_songs = len({it["song"] for it in train_ds.items})
    n_eval_songs = len({it["song"] for it in eval_ds.items})
    print(f"  train: {len(train_ds)} windows ({n_train_songs} songs), "
          f"eval: {len(eval_ds)} windows ({n_eval_songs} songs)")

    print(f"[3/4] Building model (device={device}, unfreeze_blocks={args.unfreeze_blocks})...")
    model = EEGMusicAligner(
        chs_info=egi_chs_info(),
        freeze_backbone=True,
        unfreeze_last_blocks=args.unfreeze_blocks,
    ).to(device)
    loss_fn = CLIPLoss().to(device)

    groups = model.trainable_parameters(backbone_lr_mult=args.backbone_lr_mult)
    param_groups = [{"params": g["params"], "lr": args.lr * g["lr_mult"]} for g in groups]
    param_groups.append({"params": loss_fn.parameters(), "lr": args.lr})
    opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    print(f"[4/4] Training for {args.epochs} epochs...")
    best_r1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for eeg, target, _ in train_loader:
            eeg, target = eeg.to(device), target.to(device)
            loss = loss_fn(model(eeg), target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        recalls = run_eval(model, eval_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **recalls,
               "logit_scale": float(loss_fn.logit_scale.exp().detach())}
        history.append(row)
        print(f"  epoch {epoch:3d} | loss {row['loss']:.4f} | "
              f"R@1 {recalls['R@1']:.3f} R@5 {recalls['R@5']:.3f} | "
              f"T {row['logit_scale']:.1f} | {time.time() - t0:.0f}s")

        if recalls["R@1"] > best_r1:
            best_r1 = recalls["R@1"]
            torch.save({
                "model": model.state_dict(),
                "loss": loss_fn.state_dict(),
                "epoch": epoch,
                "recalls": recalls,
                "args": vars(args),
            }, os.path.join(OUTPUT_DIR, "best_aligner.pt"))

    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest eval R@1: {best_r1:.3f}. Checkpoint: {os.path.join(OUTPUT_DIR, 'best_aligner.pt')}")


if __name__ == "__main__":
    main()
