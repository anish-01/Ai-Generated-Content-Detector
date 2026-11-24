"""Evaluate ResNet-50 and the CLIP linear probe on the in-distribution test set (issue #13).

Both models are run on exactly the same images (splits/test.csv) and scored with
accuracy and ROC-AUC. The cross-generator holdout (splits/cross_generator_test.csv)
is refused here; it belongs to the cross-generator evaluation.

Labels follow the rest of the project: 0 = real, 1 = fake. ROC-AUC is computed
from the continuous P(fake) = softmax(logits)[:, 1], never from hard labels.

A model whose checkpoint is missing is recorded as "blocked" with empty metrics;
no numbers are produced for a model that was not actually run.

Outputs:
    results/in_distribution_evaluation.csv    one row per model (metrics + status)
    results/in_distribution_predictions.csv   one row per (image, model)
    results/in_distribution_roc_curves.csv    ROC curve points per evaluated model

Usage:
    python scripts/evaluate_in_distribution.py
    python scripts/evaluate_in_distribution.py --models resnet50 --batch-size 128
    python scripts/evaluate_in_distribution.py --clip-checkpoint path/to/clip_linear_best.pth
    python scripts/evaluate_in_distribution.py --limit 256 --results-dir /tmp/smoke   # smoke test
"""

import argparse
import csv
import hashlib
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dataset import RealFakeDataset, get_device, make_loader, read_index  # noqa: E402
from models.resnet50 import LABELS, build_resnet50  # noqa: E402

MODELS = ("resnet50", "clip_linear")
PREPROCESS = {"resnet50": "resnet", "clip_linear": "clip"}  # RealFakeDataset normalization
HOLDOUT_SPLIT = "cross_generator_test"
EXPECTED_LABEL_MAP = {0: "real", 1: "fake"}

SUMMARY_FIELDS = ["model", "status", "test_images", "real_images", "fake_images", "accuracy",
                  "roc_auc", "checkpoint", "checkpoint_md5", "checkpoint_epoch", "architecture",
                  "test_manifest", "test_fingerprint", "device", "notes"]
PRED_FIELDS = ["image_path", "model", "label", "pred_label", "p_fake"]
ROC_FIELDS = ["model", "fpr", "tpr", "threshold"]


class CheckpointError(RuntimeError):
    """The checkpoint is missing or does not match the expected model."""


# ---------------------------------------------------------------- test set

def load_test_rows(csv_path, check_files=True):
    """Read the in-distribution test manifest and validate it.
    Refuses the cross-generator holdout and any row not in the 'test' split."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"test manifest not found: {csv_path} "
                                "(run: python scripts/split_datasets.py, see README)")
    if HOLDOUT_SPLIT in csv_path.name:
        raise ValueError(f"{csv_path.name} is the cross-generator holdout, not the "
                         "in-distribution test set; use splits/test.csv")
    rows = read_index(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} has no rows")

    splits = Counter(r.get("split", "test") for r in rows)
    if set(splits) != {"test"}:
        raise ValueError(f"{csv_path} contains non-test rows: {dict(splits)}")
    bad = [r["path"] for r in rows if r["label"] not in (0, 1)]
    if bad:
        raise ValueError(f"{len(bad)} rows have a label other than 0/1, e.g. {bad[0]}")
    paths = [r["path"] for r in rows]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{csv_path} lists {len(paths) - len(set(paths))} duplicate image paths")
    if {r["label"] for r in rows} != {0, 1}:
        raise ValueError("test set must contain both REAL (0) and FAKE (1) images")
    if check_files:
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(rows)} test images are missing, e.g. {missing[0]}. "
                "Download CIFAKE into data/cifake and GenImage into data/genimage (see README).")
    return rows


def subsample(rows, n, seed):
    """Seeded subset for smoke tests, always with both classes."""
    if not n or n >= len(rows):
        return rows
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    rng = random.Random(seed)
    half = n // 2
    picked = rng.sample(by_label[0], min(half, len(by_label[0])))
    picked += rng.sample(by_label[1], min(n - len(picked), len(by_label[1])))
    return sorted(picked, key=lambda r: r["path"])


def rows_fingerprint(rows):
    """MD5 over the ordered (path, label) list: identical for both models only
    if they saw exactly the same images in the same order."""
    h = hashlib.md5()
    for r in rows:
        h.update(f"{rel(r['path'])}\t{r['label']}\n".encode())
    return h.hexdigest()


def rel(p):
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def file_md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- checkpoints

def _load_ckpt(path, kind):
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"{kind} checkpoint not found: {rel(path)}")
    if path.is_dir():
        raise CheckpointError(f"{kind} checkpoint {rel(path)} is a directory, expected a .pth file")
    try:
        # plain tensors / dicts / numbers only -- no arbitrary pickled code
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise CheckpointError(f"could not load {kind} checkpoint {rel(path)}: {type(e).__name__}: {e}") from e


def _check_label_map(ckpt, path):
    lm = {int(k): str(v).lower() for k, v in (ckpt.get("label_map") or {}).items()}
    if lm != EXPECTED_LABEL_MAP:
        raise CheckpointError(f"{rel(path)}: label_map {ckpt.get('label_map')} != {EXPECTED_LABEL_MAP}")


def load_resnet50(path, device="cpu"):
    """Rebuild the trained ResNet-50 from checkpoint and validate it.
    Returns (model in eval mode, checkpoint dict)."""
    ckpt = _load_ckpt(path, "ResNet-50")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise CheckpointError(f"{rel(path)}: no 'model_state' in checkpoint")
    if ckpt.get("architecture") != "resnet50":
        raise CheckpointError(f"{rel(path)}: architecture is {ckpt.get('architecture')!r}, expected 'resnet50'")
    if ckpt.get("num_classes", 2) != 2:
        raise CheckpointError(f"{rel(path)}: num_classes is {ckpt.get('num_classes')}, expected 2")
    _check_label_map(ckpt, path)
    fc = ckpt["model_state"].get("fc.weight")
    if fc is None or tuple(fc.shape) != (2, 2048):
        raise CheckpointError(f"{rel(path)}: fc.weight shape {None if fc is None else tuple(fc.shape)}, "
                              "expected (2, 2048)")
    model = build_resnet50(pretrained=False, num_classes=2)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as e:
        raise CheckpointError(f"{rel(path)}: weights don't match the ResNet-50 architecture: {e}") from e
    return model.to(device).eval(), ckpt


def load_clip(path, device="cpu"):
    """Rebuild frozen CLIP + the trained linear head and validate it.
    Returns (model in eval mode, checkpoint dict)."""
    ckpt = _load_ckpt(path, "CLIP linear")
    if not isinstance(ckpt, dict) or "head_state" not in ckpt:
        raise CheckpointError(f"{rel(path)}: no 'head_state' in checkpoint (not a clip_linear checkpoint)")
    if ckpt.get("architecture") != "clip_linear":
        raise CheckpointError(f"{rel(path)}: architecture is {ckpt.get('architecture')!r}, expected 'clip_linear'")
    _check_label_map(ckpt, path)
    w = ckpt["head_state"].get("weight")
    if w is None or w.shape[0] != 2 or w.shape[1] != ckpt.get("embed_dim", w.shape[1]):
        raise CheckpointError(f"{rel(path)}: head weight shape {None if w is None else tuple(w.shape)} "
                              f"doesn't match 2 x embed_dim ({ckpt.get('embed_dim')})")
    try:
        from models.clip_linear import CLIPLinear  # needs open_clip_torch
    except ImportError as e:
        raise CheckpointError(f"open_clip is not installed ({e}); run: pip install open_clip_torch") from e
    model = CLIPLinear(ckpt["arch"], ckpt["pretrained"])
    if model.embed_dim != w.shape[1]:
        raise CheckpointError(f"{rel(path)}: head expects {w.shape[1]}-dim features but "
                              f"{ckpt['arch']} gives {model.embed_dim}")
    model.head.load_state_dict(ckpt["head_state"], strict=True)
    return model.to(device).eval(), ckpt


LOADERS = {"resnet50": load_resnet50, "clip_linear": load_clip}


# ---------------------------------------------------------------- inference + metrics

@torch.inference_mode()
def predict(model, rows, preprocess, device, batch_size=64, workers=4, log_every=50):
    """Run the model over rows in manifest order.
    Returns (labels, pred_labels, p_fake) as numpy arrays, one entry per row."""
    model.eval()
    ds = RealFakeDataset(rows, model=preprocess, train=False)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    labels, preds, probs = [], [], []
    start, seen = time.time(), 0
    for i, (x, y) in enumerate(loader, 1):
        logits = model(x.to(device))
        if logits.ndim != 2 or logits.shape != (x.shape[0], 2):
            raise RuntimeError(f"model output shape {tuple(logits.shape)}, expected ({x.shape[0]}, 2)")
        logits = logits.float().cpu()
        labels.append(y)
        preds.append(logits.argmax(1))
        probs.append(torch.softmax(logits, dim=1)[:, 1])
        seen += x.shape[0]
        if log_every and (i % log_every == 0 or i == len(loader)):
            rate = seen / (time.time() - start)
            print(f"    [{seen:,}/{len(rows):,}] {rate:.0f} img/s", flush=True)
    labels = torch.cat(labels).numpy().astype(int)
    preds = torch.cat(preds).numpy().astype(int)
    probs = torch.cat(probs).numpy().astype(np.float64)
    if len(probs) != len(rows):
        raise RuntimeError(f"got {len(probs)} predictions for {len(rows)} test images")
    expected = np.array([r["label"] for r in rows])
    if not np.array_equal(labels, expected):
        raise RuntimeError("loader labels don't match the manifest order")
    return labels, preds, probs


def compute_metrics(y_true, p_fake, y_pred=None):
    """Accuracy + ROC-AUC. ROC-AUC uses the continuous P(fake) scores.
    y_pred defaults to P(fake) >= 0.5 (same as argmax for 2 logits)."""
    y_true = np.asarray(y_true).astype(int)
    p_fake = np.asarray(p_fake, dtype=np.float64)
    if y_true.shape != p_fake.shape or y_true.ndim != 1:
        raise ValueError(f"shape mismatch: labels {y_true.shape}, scores {p_fake.shape}")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError(f"need both REAL (0) and FAKE (1) labels, got {sorted(set(y_true.tolist()))}")
    if not np.all(np.isfinite(p_fake)):
        raise ValueError("P(fake) scores contain NaN/inf")
    if p_fake.min() < 0 or p_fake.max() > 1:
        raise ValueError(f"P(fake) scores outside [0, 1]: [{p_fake.min()}, {p_fake.max()}]")
    if y_pred is None:
        y_pred = (p_fake >= 0.5).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if not set(np.unique(y_pred)) <= {0, 1}:
        raise ValueError("predicted labels must be 0 (real) or 1 (fake)")
    fpr, tpr, thr = roc_curve(y_true, p_fake, pos_label=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, p_fake)),
        "roc": (fpr, tpr, thr),
    }


def per_source_accuracy(rows, y_true, y_pred):
    out = defaultdict(lambda: [0, 0])
    for r, t, p in zip(rows, y_true, y_pred):
        key = f"{r.get('source_dataset', '?')}/{r.get('generator', '?')}"
        out[key][0] += int(t == p)
        out[key][1] += 1
    return {k: (c / n, n) for k, (c, n) in sorted(out.items())}


# ---------------------------------------------------------------- output

def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def fmt(x, nd=4):
    return "-" if x in (None, "") else f"{x:.{nd}f}"


def print_table(summary):
    print(f"\n{'model':12s} {'status':9s} {'images':>7s} {'accuracy':>9s} {'ROC-AUC':>8s}  checkpoint")
    for s in summary:
        print(f"{s['model']:12s} {s['status']:9s} {s['test_images'] or '-':>7} "
              f"{fmt(s['accuracy']):>9s} {fmt(s['roc_auc']):>8s}  {s['checkpoint']}")


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test-csv", type=Path, default=ROOT / "splits" / "test.csv")
    ap.add_argument("--resnet-checkpoint", type=Path, default=ROOT / "checkpoints" / "resnet50_best.pth")
    ap.add_argument("--clip-checkpoint", type=Path, default=ROOT / "checkpoints" / "clip_linear_best.pth")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--limit", type=int, help="seeded class-balanced subset (smoke tests only)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"device: {device}   torch {torch.__version__}")

    try:
        rows = load_test_rows(args.test_csv)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    rows = subsample(rows, args.limit, args.seed)
    fp = rows_fingerprint(rows)
    n_real = sum(r["label"] == 0 for r in rows)
    print(f"test set: {rel(args.test_csv)}  {len(rows):,} images "
          f"({n_real:,} real / {len(rows) - n_real:,} fake)  fingerprint {fp[:12]}"
          + ("  [SMOKE SUBSET]" if args.limit else ""))

    ckpt_paths = {"resnet50": args.resnet_checkpoint, "clip_linear": args.clip_checkpoint}
    summary, predictions, roc_rows, fingerprints = [], [], [], {}
    for name in args.models:
        path = ckpt_paths[name]
        entry = dict.fromkeys(SUMMARY_FIELDS, "")
        entry.update(model=name, checkpoint=rel(path), test_manifest=rel(args.test_csv), device=str(device))
        print(f"\n== {name}: {rel(path)}")
        try:
            model, ckpt = LOADERS[name](path, device)
        except CheckpointError as e:
            print(f"  BLOCKED: {e}")
            entry.update(status="blocked", notes=str(e))
            summary.append(entry)
            continue

        entry.update(checkpoint_md5=file_md5(path), checkpoint_epoch=ckpt.get("epoch", ""),
                     architecture=ckpt.get("arch", ckpt.get("architecture", "")))
        print(f"  checkpoint ok: {entry['architecture']}, epoch {entry['checkpoint_epoch'] or '-'}, "
              f"label_map {ckpt['label_map']}")
        t0 = time.time()
        y, yhat, p = predict(model, rows, PREPROCESS[name], device, args.batch_size, args.workers)
        m = compute_metrics(y, p, yhat)
        fingerprints[name] = rows_fingerprint(rows)
        print(f"  {len(y):,} images in {time.time() - t0:.0f}s   "
              f"accuracy {m['accuracy']:.4f}   ROC-AUC {m['roc_auc']:.4f}")
        for k, (acc, n) in per_source_accuracy(rows, y, yhat).items():
            print(f"    {k:20s} acc {acc:.4f}  (n={n:,})")

        entry.update(status="smoke" if args.limit else "evaluated", test_images=len(y),
                     real_images=int((y == 0).sum()), fake_images=int((y == 1).sum()),
                     accuracy=round(m["accuracy"], 6), roc_auc=round(m["roc_auc"], 6),
                     test_fingerprint=fingerprints[name])
        summary.append(entry)
        predictions += [{"image_path": rel(r["path"]), "model": name, "label": int(t),
                         "pred_label": int(q), "p_fake": f"{pf:.8f}"}
                        for r, t, q, pf in zip(rows, y, yhat, p)]
        fpr, tpr, thr = m["roc"]
        roc_rows += [{"model": name, "fpr": f"{a:.6f}", "tpr": f"{b:.6f}",
                      "threshold": f"{c:.8f}" if math.isfinite(c) else "inf"}
                     for a, b, c in zip(fpr, tpr, thr)]
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    if len(set(fingerprints.values())) > 1:
        sys.exit("error: models were evaluated on different test images")

    out = args.results_dir
    write_csv(out / "in_distribution_evaluation.csv", SUMMARY_FIELDS, summary)
    write_csv(out / "in_distribution_predictions.csv", PRED_FIELDS, predictions)
    write_csv(out / "in_distribution_roc_curves.csv", ROC_FIELDS, roc_rows)
    print_table(summary)
    print(f"\nwrote {rel(out / 'in_distribution_evaluation.csv')}, "
          f"{rel(out / 'in_distribution_predictions.csv')} ({len(predictions):,} rows), "
          f"{rel(out / 'in_distribution_roc_curves.csv')}")

    blocked = [s["model"] for s in summary if s["status"] == "blocked"]
    if blocked:
        print(f"\nNOT EVALUATED: {', '.join(blocked)} (see notes in the results CSV)")
    return 1 if len(blocked) == len(summary) else 0


if __name__ == "__main__":
    sys.exit(main())
