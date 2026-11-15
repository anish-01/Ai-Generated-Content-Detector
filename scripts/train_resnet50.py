"""Fine-tune ImageNet ResNet-50 as a real (0) vs generated (1) classifier (issue #9).

Trains on splits/train.csv and picks the best epoch on splits/val.csv (lowest
val loss). The test split and the cross-generator holdout are never touched.

Outputs:
    checkpoints/resnet50_best.pth   best val loss so far (git-ignored)
    checkpoints/resnet50_last.pth   end of the latest epoch, for --resume
    results/resnet50_training_history.csv
    results/resnet50_loss.png, results/resnet50_accuracy.png

Defaults (fine-tuning the whole network):
    AdamW, lr 1e-4, weight decay 1e-4 -- small lr so the pretrained features
    aren't wrecked early on; AdamW handles the new fc layer and the backbone
    without per-layer tuning.
    Linear warmup (first 3% of steps) then cosine decay to 1% of the lr.
    Batch 32 -- fits in 8 GB of unified memory on an M1.

Usage:
    python scripts/train_resnet50.py                       # 3 epochs
    python scripts/train_resnet50.py --epochs 5 --batch-size 64
    python scripts/train_resnet50.py --resume              # continue from resnet50_last.pth
    python scripts/train_resnet50.py --resume --epochs 5   # continue and extend to 5 epochs
    python scripts/train_resnet50.py --limit-train 256 --limit-val 128 --epochs 1   # smoke test
"""

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dataset import RealFakeDataset, get_device, read_index  # noqa: E402
from models.resnet50 import WEIGHTS, build_resnet50, load_checkpoint, save_checkpoint  # noqa: E402

HISTORY_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=ROOT / "splits" / "train.csv")
    ap.add_argument("--val", type=Path, default=ROOT / "splits" / "val.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03, help="fraction of steps for linear warmup")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--resume", nargs="?", const="last", default=None,
                    help="resume from checkpoints/resnet50_last.pth (or a given path)")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--limit-train", type=int, help="use a random subset (smoke tests)")
    ap.add_argument("--limit-val", type=int, help="use a random subset (smoke tests)")
    ap.add_argument("--log-every", type=int, default=100, help="batches between progress lines")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    return ap.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_rows(csv_path, split, limit, seed):
    rows = [r for r in read_index(csv_path) if r.get("split", split) == split]
    if not rows:
        sys.exit(f"no '{split}' rows in {csv_path}")
    missing = [r["path"] for r in rows if not Path(r["path"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} images listed in {csv_path} are missing, e.g. {missing[0]} "
                 "(regenerate the data: see README)")
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def make_loaders(args, device):
    train_rows = load_rows(args.train, "train", args.limit_train, args.seed)
    val_rows = load_rows(args.val, "val", args.limit_val, args.seed)
    train_ds = RealFakeDataset(train_rows, model="resnet", train=True)
    val_ds = RealFakeDataset(val_rows, model="resnet", train=False)
    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, generator=gen, **common)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def make_scheduler(optimizer, total_steps, warmup_frac, floor=0.01):
    """Linear warmup then cosine decay to floor * lr. A LambdaLR is recomputed
    from the step count, so it can be rebuilt with a new total on --resume."""
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, log_every):
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    start = time.time()
    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        n = y.size(0)
        loss_sum += loss.item() * n
        correct += (logits.argmax(1) == y).sum().item()
        seen += n
        if i % log_every == 0 or i == len(loader):
            rate = seen / (time.time() - start)
            eta = (len(loader.dataset) - seen) / rate
            print(f"  epoch {epoch} [{i}/{len(loader)}] loss {loss_sum / seen:.4f} "
                  f"acc {correct / seen:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                  f"{rate:.0f} img/s eta {eta / 60:.1f} min", flush=True)
    return loss_sum / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    return loss_sum / seen, correct / seen


def write_history(history, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "resnet50_training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        w.writerows(history)
    plot_history(history, results_dir)


def plot_history(history, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    for metric, ylabel in (("loss", "cross-entropy loss"), ("acc", "accuracy")):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, [h[f"train_{metric}"] for h in history], "o-", label="train")
        ax.plot(epochs, [h[f"val_{metric}"] for h in history], "s-", label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"ResNet-50 training vs validation {'loss' if metric == 'loss' else 'accuracy'}")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / f"resnet50_{'loss' if metric == 'loss' else 'accuracy'}.png", dpi=120)
        plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"device: {device}   torch {torch.__version__}")

    train_loader, val_loader = make_loaders(args, device)
    print(f"train: {len(train_loader.dataset):,} images   val: {len(val_loader.dataset):,} images   "
          f"batch {args.batch_size}   {len(train_loader)} steps/epoch")

    model = build_resnet50(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs * len(train_loader), args.warmup)

    start_epoch, best_val_loss, best_epoch, history = 1, float("inf"), 0, []
    if args.resume:
        path = args.checkpoint_dir / "resnet50_last.pth" if args.resume == "last" else Path(args.resume)
        model, ckpt = load_checkpoint(path, model, optimizer, scheduler, device)
        start_epoch = ckpt["epoch"] + 1
        best_val_loss, best_epoch = ckpt["best_val_loss"], ckpt["best_epoch"]
        history = ckpt.get("history", [])
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        print(f"resumed from {path} after epoch {ckpt['epoch']} "
              f"(best val loss {best_val_loss:.4f} at epoch {best_epoch})")
        if start_epoch > args.epochs:
            print(f"already trained {ckpt['epoch']} epochs; pass --epochs N with N > {ckpt['epoch']} to continue")
            return

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=str(device), weights=str(WEIGHTS) if not args.no_pretrained else None,
                  train_images=len(train_loader.dataset), val_images=len(val_loader.dataset))

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                scheduler, device, epoch, args.log_every)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "train_loss": train_loss,
               "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
               "seconds": round(time.time() - t0, 1)}
        history.append(row)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, best_epoch = val_loss, epoch
        state = dict(epoch=epoch, metrics=row, best_val_loss=best_val_loss, best_epoch=best_epoch,
                     history=history, config=config, torch_rng=torch.get_rng_state())
        save_checkpoint(args.checkpoint_dir / "resnet50_last.pth", model, optimizer, scheduler, **state)
        if improved:
            save_checkpoint(args.checkpoint_dir / "resnet50_best.pth", model, optimizer, scheduler, **state)
        write_history(history, args.results_dir)

        print(f"epoch {epoch}/{args.epochs}  train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {row['lr']:.2e} | "
              f"{row['seconds'] / 60:.1f} min" + ("  * best, saved" if improved else ""), flush=True)

    print(f"done. best val loss {best_val_loss:.4f} at epoch {best_epoch} -> "
          f"{args.checkpoint_dir / 'resnet50_best.pth'}")


if __name__ == "__main__":
    main()
