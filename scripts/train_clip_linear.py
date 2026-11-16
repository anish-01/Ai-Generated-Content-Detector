"""Train the frozen-CLIP linear classifier (issue #10).

Two steps:
  1. Extract: run every train/val image through the frozen CLIP image encoder
     once and cache the embeddings in features/. This is the slow part
     (~12 img/s for ViT-L/14 on an M2), so it saves progress as it goes --
     if it gets interrupted just run the same command again and it picks up
     where it stopped.
  2. Fit: train a logistic regression on the cached embeddings for a range
     of regularization strengths C, and keep the one with the lowest val loss
     (same rule the ResNet script uses). Takes seconds.

Like the ResNet script, the test split and the cross-generator holdout are
never touched here -- that's for the eval tasks.

Outputs:
    features/<arch>_<pretrained>_{train,val}.pt    cached embeddings (git-ignored)
    checkpoints/clip_linear_best.pth               the trained head (git-ignored)
    results/clip_linear_c_sweep.csv                val metrics for every C
    results/clip_linear_c_sweep.png                val loss / accuracy vs C

Usage:
    python scripts/train_clip_linear.py                       # ViT-L/14, ~2.5 h to extract on an M2
    python scripts/train_clip_linear.py --arch ViT-B-16       # ~30 min, a bit weaker
    python scripts/train_clip_linear.py --balance-sources     # weight CIFAKE and GenImage equally
    python scripts/train_clip_linear.py --limit-train 256 --limit-val 128   # smoke test
"""

import argparse
import csv
import hashlib
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.clip_linear import (DEFAULT_ARCH, DEFAULT_PRETRAINED, CLIPLinear,  # noqa: E402
                                head_from_logistic, save_checkpoint)
from models.dataset import RealFakeDataset, get_device, read_index  # noqa: E402

SWEEP_FIELDS = ["C", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc", "seconds"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=ROOT / "splits" / "train.csv")
    ap.add_argument("--val", type=Path, default=ROOT / "splits" / "val.csv")
    ap.add_argument("--arch", default=DEFAULT_ARCH, help="open_clip model name, e.g. ViT-L-14, ViT-B-16")
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    ap.add_argument("--C", type=float, nargs="+", default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                    help="inverse regularization strengths to try (smaller = more regularized)")
    ap.add_argument("--balance-sources", action="store_true",
                    help="weight samples so each source_dataset counts equally (GenImage is tiny next to CIFAKE)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=50, help="batches between partial cache saves")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--features-dir", type=Path, default=ROOT / "features")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--limit-train", type=int, help="use a random subset (smoke tests)")
    ap.add_argument("--limit-val", type=int, help="use a random subset (smoke tests)")
    ap.add_argument("--extract-only", action="store_true", help="just build the feature cache")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- data

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


def rows_key(rows):
    """Fingerprint of the exact image list, so a stale cache never gets reused."""
    h = hashlib.md5()
    for r in rows:
        h.update(r["path"].encode())
    return h.hexdigest()


# ---------------------------------------------------------------- step 1: extract

@torch.no_grad()
def extract(model, rows, cache, args, device, name):
    key = rows_key(rows)
    if cache.exists():
        c = torch.load(cache, weights_only=False)
        if c["key"] == key:
            print(f"{name}: using cached features {cache.relative_to(ROOT)} {tuple(c['feats'].shape)}")
            return c["feats"]
        print(f"{name}: cache {cache.name} is for a different image list, re-extracting")

    partial = cache.with_suffix(".partial.pt")
    chunks, done = [], 0
    if partial.exists():
        p = torch.load(partial, weights_only=False)
        if p["key"] == key:
            chunks, done = [p["feats"]], p["done"]
            print(f"{name}: resuming from {done:,}/{len(rows):,}")

    ds = RealFakeDataset(rows[done:], model="clip", train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers,
                                         persistent_workers=args.workers > 0)
    start, seen = time.time(), 0
    for i, (x, _) in enumerate(loader, 1):
        chunks.append(model.embed(x.to(device)).cpu())
        seen += x.size(0)
        if i % args.save_every == 0 or i == len(loader):
            feats = torch.cat(chunks)
            chunks = [feats]
            torch.save({"key": key, "feats": feats, "done": done + seen}, partial)
            rate = seen / (time.time() - start)
            eta = (len(rows) - done - seen) / rate
            print(f"  {name} [{done + seen:,}/{len(rows):,}] {rate:.1f} img/s "
                  f"eta {eta / 60:.1f} min", flush=True)

    feats = torch.cat(chunks)
    assert feats.shape[0] == len(rows), (feats.shape, len(rows))
    torch.save({"key": key, "feats": feats, "arch": args.arch, "pretrained": args.pretrained,
                "paths": [r["path"] for r in rows]}, cache)
    partial.unlink(missing_ok=True)
    print(f"{name}: saved {cache.relative_to(ROOT)} {tuple(feats.shape)}")
    return feats


# ---------------------------------------------------------------- step 2: fit

def source_weights(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r.get("source_dataset", "?")] += 1
    n, k = len(rows), len(counts)
    return np.array([n / (k * counts[r.get("source_dataset", "?")]) for r in rows])


def per_source_acc(rows, y_true, y_pred):
    out = defaultdict(lambda: [0, 0])
    for r, t, p in zip(rows, y_true, y_pred):
        key = f"{r.get('source_dataset', '?')}/{r.get('generator', '?')}"
        out[key][0] += int(t == p)
        out[key][1] += 1
    return {k: (c / n, n) for k, (c, n) in sorted(out.items())}


def fit_sweep(Xtr, ytr, Xva, yva, weights, args):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
    sweep, best = [], None
    for C in args.C:
        t0 = time.time()
        clf = LogisticRegression(C=C, max_iter=5000, random_state=args.seed)
        clf.fit(Xtr_s, ytr, sample_weight=weights)
        ptr, pva = clf.predict_proba(Xtr_s)[:, 1], clf.predict_proba(Xva_s)[:, 1]
        row = {"C": C,
               "train_loss": log_loss(ytr, ptr), "train_acc": accuracy_score(ytr, ptr > 0.5),
               "val_loss": log_loss(yva, pva), "val_acc": accuracy_score(yva, pva > 0.5),
               "val_auc": roc_auc_score(yva, pva), "seconds": round(time.time() - t0, 1)}
        sweep.append(row)
        flag = ""
        if best is None or row["val_loss"] < best[0]["val_loss"]:
            best, flag = (row, clf, pva), "  * best"
        print(f"  C={C:<6g} train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
              f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} auc {row['val_auc']:.4f}"
              f"{flag}", flush=True)
    return scaler, sweep, best


def fold_scaler(clf, scaler):
    """LR was trained on standardized features (x - mean) / std. Fold that into
    the weights so the saved head works directly on raw CLIP embeddings."""
    w = clf.coef_[0] / scaler.scale_
    b = clf.intercept_[0] - np.dot(w, scaler.mean_)
    return w, b


def write_results(sweep, best_C, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "clip_linear_c_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        w.writeheader()
        w.writerows(sweep)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Cs = [r["C"] for r in sweep]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in ((axes[0], "loss", "log loss"), (axes[1], "acc", "accuracy")):
        ax.plot(Cs, [r[f"train_{metric}"] for r in sweep], "o-", label="train")
        ax.plot(Cs, [r[f"val_{metric}"] for r in sweep], "s-", label="validation")
        ax.axvline(best_C, color="gray", ls="--", lw=1, label=f"picked C={best_C:g}")
        ax.set_xscale("log")
        ax.set_xlabel("C (inverse regularization)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("CLIP linear probe: regularization sweep")
    fig.tight_layout()
    fig.savefig(results_dir / "clip_linear_c_sweep.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"device: {device}   torch {torch.__version__}   CLIP {args.arch}/{args.pretrained}")

    train_rows = load_rows(args.train, "train", args.limit_train, args.seed)
    val_rows = load_rows(args.val, "val", args.limit_val, args.seed)
    print(f"train: {len(train_rows):,} images   val: {len(val_rows):,} images")

    model = CLIPLinear(args.arch, args.pretrained).to(device).eval()
    args.features_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arch}_{args.pretrained}"
    suffix = "_smoke" if (args.limit_train or args.limit_val) else ""
    Xtr = extract(model, train_rows, args.features_dir / f"{tag}_train{suffix}.pt", args, device, "train")
    Xva = extract(model, val_rows, args.features_dir / f"{tag}_val{suffix}.pt", args, device, "val")
    if args.extract_only:
        return

    ytr = np.array([r["label"] for r in train_rows])
    yva = np.array([r["label"] for r in val_rows])
    weights = source_weights(train_rows) if args.balance_sources else None
    print(f"\nfitting logistic regression on {Xtr.shape[1]}-dim features"
          f"{' (sources balanced)' if weights is not None else ''}")
    scaler, sweep, (best, clf, pva) = fit_sweep(Xtr.numpy(), ytr, Xva.numpy(), yva, weights, args)

    w, b = fold_scaler(clf, scaler)
    model.head = head_from_logistic(w, b).to(device)
    # sanity check: the torch head on raw features must match sklearn
    with torch.no_grad():
        p_head = model.head(Xva.to(device)).softmax(1)[:, 1].cpu().numpy()
    gap = float(np.abs(p_head - pva).max())
    assert gap < 1e-3, f"head doesn't match sklearn (max diff {gap})"

    by_source = per_source_acc(val_rows, yva, pva > 0.5)
    print(f"\nbest C={best['C']:g}: val loss {best['val_loss']:.4f} acc {best['val_acc']:.4f} "
          f"auc {best['val_auc']:.4f}")
    for k, (acc, n) in by_source.items():
        print(f"  val {k:22s} acc {acc:.4f}  (n={n:,})")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=str(device), train_images=len(train_rows), val_images=len(val_rows))
    ckpt = args.checkpoint_dir / f"clip_linear_best{suffix}.pth"
    save_checkpoint(ckpt, model, C=best["C"], metrics=best, sweep=sweep,
                    val_by_source={k: v[0] for k, v in by_source.items()}, config=config)
    if not suffix:
        write_results(sweep, best["C"], args.results_dir)
    print(f"saved {ckpt.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
