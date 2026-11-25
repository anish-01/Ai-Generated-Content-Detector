"""Robustness of the trained detector to JPEG compression, resizing and cropping (issue #14).

Every distortion is applied in memory to the *original* test image (never to an
already-distorted one), before the model's normal preprocessing
(Resize(224, bicubic) -> CenterCrop(224) -> ToTensor -> Normalize). The test set,
checkpoint validation, inference loop and metrics are reused from
scripts/evaluate_in_distribution.py (issue #13), so the clean row here is directly
comparable to the issue #13 result.

Distortion families (all deterministic, all bicubic resampling as in the project
preprocessing):
    clean            no distortion -- re-measured baseline
    jpeg             JPEG encode/decode in memory at quality 90/70/50/30/10
                     (Pillow encoder, default 4:2:0 chroma subsampling)
    resize           original -> S x S -> back to original size, S = 192/128/64/32
                     (literal issue spec; images already <= S are upsampled, so the
                     round trip removes little information for them)
    resize_relative  supplementary: original -> scaled by S/224 -> back to original,
                     i.e. the detail of an S-pixel image on the 224-pixel model input.
                     Unlike `resize`, this degrades the 32x32 CIFAKE images too.
    crop             centred crop keeping 90/75/50/25 % of the image area (each side
                     scaled by sqrt(area)), resized back to the original size

Results are saved after every completed setting, and a re-run skips settings that
are already done for the same checkpoint + test set, so the evaluation can be
interrupted and resumed. Accuracy drops are measured against the *re-measured*
clean baseline:  accuracy_drop = baseline_accuracy - distorted_accuracy
(negative = accuracy went up).

Outputs:
    results/robustness_evaluation.csv            one row per (model, distortion, level)
    results/robustness_predictions/*.csv.gz      per-image predictions per setting
    results/robustness_charts/*.png              charts built from the CSV

Usage:
    python scripts/evaluate_robustness.py                         # everything, resumable
    python scripts/evaluate_robustness.py --distortions clean jpeg
    python scripts/evaluate_robustness.py --model clip_linear --clip-checkpoint path.pth
    python scripts/evaluate_robustness.py --verify                # recheck CSV vs predictions
    python scripts/evaluate_robustness.py --charts-only
    python scripts/evaluate_robustness.py --limit 256             # smoke test -> results/robustness_smoke
"""

import argparse
import csv
import gzip
import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_in_distribution as ev  # noqa: E402  (issue #13 utilities)
from models.dataset import IMG_SIZE, get_device  # noqa: E402

RESAMPLE = Image.BICUBIC
RESAMPLE_NAME = "bicubic"

# canonical order: clean first (it is the baseline for every other row)
DEFAULT_LEVELS = {
    "clean": [None],
    "jpeg": [90, 70, 50, 30, 10],
    "resize": [192, 128, 64, 32],
    "resize_relative": [192, 128, 64, 32],
    "crop": [0.90, 0.75, 0.50, 0.25],
}
DISTORTIONS = tuple(DEFAULT_LEVELS)

ISSUE13_CSV = ROOT / "results" / "in_distribution_evaluation.csv"
MODEL_SELECTION_NOTE = {
    "resnet50": "provisional: CLIP linear probe not yet evaluated in-distribution (issue #13)",
    "clip_linear": "candidate: confirm against issue #13 comparison",
}

FIELDS = ["model", "checkpoint", "checkpoint_md5", "distortion", "level", "level_label",
          "num_images", "n_correct", "accuracy", "roc_auc", "real_accuracy", "fake_accuracy",
          "baseline_accuracy", "accuracy_drop", "accuracy_drop_pp", "baseline_roc_auc",
          "roc_auc_drop", "status", "model_selection", "test_manifest", "test_fingerprint",
          "device", "resample", "elapsed_s", "notes"]
PRED_FIELDS = ["row", "label", "pred_label", "p_fake"]


# ---------------------------------------------------------------- distortions
# Callable classes (not closures) so DataLoader workers can pickle them under
# macOS "spawn". Each takes and returns an RGB PIL image and never mutates input.

class Identity:
    name = "clean"

    def __call__(self, img):
        return img


class JPEGCompress:
    """Real JPEG encode + decode in memory."""

    def __init__(self, quality):
        if isinstance(quality, bool) or not isinstance(quality, (int, np.integer)) \
                or not 1 <= quality <= 100:
            raise ValueError(f"JPEG quality must be an integer in 1..100, got {quality!r}")
        self.quality = int(quality)
        self.name = f"jpeg_q{self.quality}"

    def __call__(self, img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=self.quality)
        out = Image.open(io.BytesIO(buf.getvalue()))
        out.load()
        return out.convert("RGB")


def _check_size(size, what):
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError(f"{what} must be a positive integer, got {size!r}")
    return int(size)


class ResizeRoundTrip:
    """Original -> size x size -> original size (literal issue #14 resize test)."""

    def __init__(self, size):
        self.size = _check_size(size, "resize target")
        self.name = f"resize_{self.size}"

    def __call__(self, img):
        w, h = img.size
        small = img.resize((self.size, self.size), RESAMPLE)
        return small.resize((w, h), RESAMPLE)


class RelativeResizeRoundTrip:
    """Scale both sides by size/224, then back to original size. Keeps aspect ratio."""

    def __init__(self, size, reference=IMG_SIZE):
        self.size = _check_size(size, "relative resize target")
        if self.size > reference:
            raise ValueError(f"relative resize target {self.size} exceeds model input {reference}")
        self.scale = self.size / reference
        self.name = f"resize_relative_{self.size}"

    def small_size(self, w, h):
        return max(1, round(w * self.scale)), max(1, round(h * self.scale))

    def __call__(self, img):
        w, h = img.size
        return img.resize(self.small_size(w, h), RESAMPLE).resize((w, h), RESAMPLE)


def crop_box(w, h, area):
    """Centred crop box keeping `area` of the image area (sides scaled by sqrt(area))."""
    if isinstance(area, bool) or not isinstance(area, (int, float, np.floating)) \
            or not 0 < area <= 1:
        raise ValueError(f"crop area fraction must be in (0, 1], got {area!r}")
    s = math.sqrt(area)
    cw, ch = max(1, min(w, round(w * s))), max(1, min(h, round(h * s)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return left, top, left + cw, top + ch


class CenterCropArea:
    """Centred crop keeping `area` of the pixels, resized back to the original size."""

    def __init__(self, area):
        crop_box(100, 100, area)  # validates
        self.area = float(area)
        self.name = f"crop_{round(self.area * 100)}pct"

    def __call__(self, img):
        w, h = img.size
        return img.crop(crop_box(w, h, self.area)).resize((w, h), RESAMPLE)


def make_distortion(kind, level):
    if kind == "clean":
        return Identity()
    if kind == "jpeg":
        return JPEGCompress(level)
    if kind == "resize":
        return ResizeRoundTrip(level)
    if kind == "resize_relative":
        return RelativeResizeRoundTrip(level)
    if kind == "crop":
        return CenterCropArea(level)
    raise ValueError(f"unknown distortion {kind!r}; choose from {', '.join(DISTORTIONS)}")


def level_label(kind, level):
    if kind == "clean":
        return "none"
    if kind == "jpeg":
        return f"quality {level}"
    if kind == "resize":
        return f"{level}x{level}"
    if kind == "resize_relative":
        return f"scale {level}/{IMG_SIZE}"
    if kind == "crop":
        return f"{round(level * 100)}% area"
    raise ValueError(f"unknown distortion {kind!r}")


def level_str(kind, level):
    return "" if kind == "clean" else (f"{level:.2f}" if kind == "crop" else str(level))


# ---------------------------------------------------------------- metrics

def accuracy_drop(baseline, distorted):
    """Returns (drop, drop in percentage points); negative = accuracy improved."""
    d = float(baseline) - float(distorted)
    return d, d * 100.0


def setting_metrics(y, yhat, p):
    m = ev.compute_metrics(y, p, yhat)
    y, yhat = np.asarray(y), np.asarray(yhat)
    return {
        "num_images": int(len(y)),
        "n_correct": int((y == yhat).sum()),
        "accuracy": m["accuracy"],
        "roc_auc": m["roc_auc"],
        "real_accuracy": float((yhat[y == 0] == 0).mean()),
        "fake_accuracy": float((yhat[y == 1] == 1).mean()),
    }


def distortion_notes(kind, level, sizes):
    """Describe what the distortion actually does to this test set."""
    if kind == "resize":
        up = sum(1 for w, h in sizes if w <= level and h <= level)
        if up:
            return (f"{up}/{len(sizes)} images are already <= {level}x{level}; for them the "
                    "round trip is an upsample, not a downscale")
    if kind == "crop":
        fr = [(lambda b: (b[2] - b[0]) * (b[3] - b[1]) / (w * h))(crop_box(w, h, level))
              for w, h in sizes]
        return f"realised area fraction mean {np.mean(fr):.3f} (min {min(fr):.3f}, max {max(fr):.3f})"
    if kind == "resize_relative":
        d = RelativeResizeRoundTrip(level)
        small = [d.small_size(w, h) for w, h in sizes]
        return f"intermediate size e.g. 32x32 -> {d.small_size(32, 32)[0]}x{d.small_size(32, 32)[1]}; " \
               f"smallest side min {min(min(s) for s in small)}"
    return ""


# ---------------------------------------------------------------- inference

@torch.inference_mode()
def predict(model, rows, preprocess, device, distortion, batch_size=64, workers=4, log_every=50):
    """Issue #13's predict() with a distortion applied to each original image."""
    from models.dataset import RealFakeDataset, make_loader
    model.eval()
    ds = RealFakeDataset(rows, model=preprocess, train=False, distortion=distortion)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    labels, preds, probs = [], [], []
    start, seen = time.time(), 0
    for i, (x, y) in enumerate(loader, 1):
        if x.shape[1:] != (3, IMG_SIZE, IMG_SIZE):
            raise RuntimeError(f"preprocessed batch has shape {tuple(x.shape)}")
        logits = model(x.to(device))
        if logits.shape != (x.shape[0], 2):
            raise RuntimeError(f"model output shape {tuple(logits.shape)}, expected ({x.shape[0]}, 2)")
        logits = logits.float().cpu()
        labels.append(y)
        preds.append(logits.argmax(1))
        probs.append(torch.softmax(logits, dim=1)[:, 1])
        seen += x.shape[0]
        if log_every and (i % log_every == 0 or i == len(loader)):
            print(f"    [{seen:,}/{len(rows):,}] {seen / (time.time() - start):.0f} img/s", flush=True)
    labels = torch.cat(labels).numpy().astype(int)
    preds = torch.cat(preds).numpy().astype(int)
    probs = torch.cat(probs).numpy().astype(np.float64)
    if len(probs) != len(rows):
        raise RuntimeError(f"got {len(probs)} predictions for {len(rows)} test images")
    if not np.array_equal(labels, [r["label"] for r in rows]):
        raise RuntimeError("loader labels don't match the manifest order")
    return labels, preds, probs


# ---------------------------------------------------------------- results I/O

def read_results(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and list(rows[0].keys()) != FIELDS:
        raise ValueError(f"{path} has unexpected columns; move it aside or delete it to start over")
    return rows


def order_key(r):
    kinds = list(DISTORTIONS)
    k = kinds.index(r["distortion"]) if r["distortion"] in kinds else len(kinds)
    lv = DEFAULT_LEVELS.get(r["distortion"], [])
    try:
        val = float(r["level"]) if r["level"] != "" else None
        i = next(j for j, x in enumerate(lv) if (x is None and val is None) or
                 (x is not None and val is not None and math.isclose(x, val)))
    except (StopIteration, ValueError):
        i = len(lv)
    return (r["model"], k, i, r["level"])


def upsert(rows, new):
    key = (new["model"], new["distortion"], new["level"])
    rows = [r for r in rows if (r["model"], r["distortion"], r["level"]) != key] + [new]
    return sorted(rows, key=order_key)


def pred_path(pred_dir, model, kind, level):
    tag = "clean" if kind == "clean" else make_distortion(kind, level).name
    return Path(pred_dir) / f"{model}_{tag}.csv.gz"


def write_predictions(path, y, yhat, p):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(PRED_FIELDS)
        w.writerows((i, int(a), int(b), f"{c:.9g}") for i, (a, b, c) in enumerate(zip(y, yhat, p)))
    tmp.replace(path)


def read_predictions(path):
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f))
    return (np.array([int(r["label"]) for r in rows]), np.array([int(r["pred_label"]) for r in rows]),
            np.array([float(r["p_fake"]) for r in rows]))


def issue13_reference(model):
    for r in (read_results_loose(ISSUE13_CSV)):
        if r.get("model") == model and r.get("status") == "evaluated":
            return r
    return None


def read_results_loose(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- verify

def verify(results_csv, pred_dir, tol=1e-6):
    """Recompute every row's metrics and drops from its saved predictions."""
    rows = read_results(results_csv)
    if not rows:
        print(f"nothing to verify: {results_csv} missing or empty")
        return 1
    bad = 0
    for r in rows:
        if r["status"] not in ("evaluated", "smoke"):
            continue
        level = None if r["level"] == "" else (float(r["level"]) if r["distortion"] == "crop" else int(r["level"]))
        path = pred_path(pred_dir, r["model"], r["distortion"], level)
        if not path.exists():
            print(f"  MISSING predictions {path.name}")
            bad += 1
            continue
        y, yhat, p = read_predictions(path)
        m = setting_metrics(y, yhat, p)
        d, dpp = accuracy_drop(r["baseline_accuracy"], m["accuracy"])
        checks = {"num_images": m["num_images"] == int(r["num_images"]),
                  "n_correct": m["n_correct"] == int(r["n_correct"]),
                  "accuracy": abs(m["accuracy"] - float(r["accuracy"])) < tol,
                  "accuracy=n_correct/n": abs(int(r["n_correct"]) / int(r["num_images"]) - float(r["accuracy"])) < tol,
                  "roc_auc": abs(m["roc_auc"] - float(r["roc_auc"])) < tol,
                  "accuracy_drop": abs(d - float(r["accuracy_drop"])) < tol,
                  "accuracy_drop_pp": abs(dpp - float(r["accuracy_drop_pp"])) < 1e-4}
        failed = [k for k, ok in checks.items() if not ok]
        bad += bool(failed)
        print(f"  {'FAIL ' + ','.join(failed) if failed else 'ok':28s} {r['model']:12s} "
              f"{r['distortion']:16s} {r['level_label']}")
    print(f"verify: {'all rows consistent' if not bad else f'{bad} problem(s)'}")
    return 1 if bad else 0


# ---------------------------------------------------------------- charts

def make_charts(results_csv, out_dir, model=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in read_results(results_csv) if r["status"] in ("evaluated", "smoke")
            and (model is None or r["model"] == model)]
    if not rows:
        print("no results to chart")
        return []
    model = model or rows[0]["model"]
    rows = [r for r in rows if r["model"] == model]
    base = next((r for r in rows if r["distortion"] == "clean"), None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    tag = " (provisional model)" if model == "resnet50" else ""

    def series(kind):
        s = sorted([r for r in rows if r["distortion"] == kind], key=lambda r: float(r["level"]))
        return [float(r["level"]) for r in s], [float(r["accuracy"]) * 100 for r in s], \
               [float(r["roc_auc"]) for r in s]

    def line_chart(fname, kinds, xlabel, title, xfmt=None, invert=True):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for kind, label, color in kinds:
            x, acc, _ = series(kind)
            if not x:
                continue
            plotted = True
            ax.plot(x, acc, marker="o", color=color, label=label)
            for xi, ai in zip(x, acc):
                ax.annotate(f"{ai:.1f}", (xi, ai), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=8, color=color)
        if not plotted:
            plt.close(fig)
            return
        if base:
            ax.axhline(float(base["accuracy"]) * 100, ls="--", color="gray",
                       label=f"clean baseline ({float(base['accuracy']) * 100:.2f}%)")
        if invert:
            ax.invert_xaxis()
        if xfmt:
            ax.xaxis.set_major_formatter(xfmt)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(title + tag)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    from matplotlib.ticker import FuncFormatter
    n = rows[0]["num_images"]
    line_chart("jpeg_quality_vs_accuracy.png", [("jpeg", "JPEG compressed", "#1f77b4")],
               "JPEG quality (lower = stronger compression)",
               f"{model}: JPEG quality vs accuracy (n={n})")
    line_chart("resize_vs_accuracy.png",
               [("resize", "resize to S×S and back (issue spec)", "#2ca02c"),
                ("resize_relative", f"scale by S/{IMG_SIZE} and back (supplementary)", "#d62728")],
               "Resolution S (pixels)", f"{model}: resize resolution vs accuracy (n={n})")
    line_chart("crop_area_vs_accuracy.png", [("crop", "centre crop, resized back", "#9467bd")],
               "Retained image area", f"{model}: retained crop area vs accuracy (n={n})",
               xfmt=FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))

    # drop by setting
    dist = [r for r in rows if r["distortion"] != "clean" and r["accuracy_drop_pp"] != ""]
    if dist:
        colors = {"jpeg": "#1f77b4", "resize": "#2ca02c", "resize_relative": "#d62728", "crop": "#9467bd"}
        names = {"jpeg": "JPEG", "resize": "resize", "resize_relative": "resize (rel.)", "crop": "crop"}
        labels = [f"{names[r['distortion']]} {r['level_label']}" for r in dist]
        vals = [float(r["accuracy_drop_pp"]) for r in dist]
        fig, ax = plt.subplots(figsize=(8, 0.38 * len(dist) + 1.5))
        y = np.arange(len(dist))
        ax.barh(y, vals, color=[colors[r["distortion"]] for r in dist])
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:+.2f}", (v, yi), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                        va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Accuracy drop vs clean baseline (percentage points; negative = improved)")
        ax.set_title(f"{model}: accuracy drop by distortion setting{tag}")
        ax.grid(axis="x", alpha=0.3)
        from matplotlib.patches import Patch
        present = [k for k in colors if any(r["distortion"] == k for r in dist)]
        ax.legend(handles=[Patch(color=colors[k], label=names[k]) for k in present], fontsize=8,
                  loc="upper right")
        lo, hi = min(vals + [0]), max(vals + [0])
        pad = 0.12 * (hi - lo or 1)
        ax.set_xlim(lo - pad, hi + pad)
        fig.tight_layout()
        p = out_dir / "accuracy_drop_by_setting.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
    for p in written:
        print(f"  chart: {ev.rel(p)}")
    return written


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=ev.MODELS, default="resnet50")
    ap.add_argument("--resnet-checkpoint", type=Path, default=ROOT / "checkpoints" / "resnet50_best.pth")
    ap.add_argument("--clip-checkpoint", type=Path, default=ROOT / "checkpoints" / "clip_linear_best.pth")
    ap.add_argument("--test-csv", type=Path, default=ROOT / "splits" / "test.csv")
    ap.add_argument("--distortions", nargs="+", choices=DISTORTIONS, default=list(DISTORTIONS),
                    help="families to run (clean is always run first if its result is missing)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--results-dir", type=Path, help="default results/ (results/robustness_smoke with --limit)")
    ap.add_argument("--limit", type=int, help="seeded class-balanced subset (smoke tests only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="re-run settings that already have results")
    ap.add_argument("--verify", action="store_true", help="recheck the CSV against saved predictions and exit")
    ap.add_argument("--charts-only", action="store_true", help="rebuild charts from the CSV and exit")
    args = ap.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        ap.error("--batch-size must be >= 1 and --workers >= 0")
    if args.limit is not None and args.limit < 2:
        ap.error("--limit must be >= 2")
    if args.results_dir is None:
        args.results_dir = ROOT / "results" / ("robustness_smoke" if args.limit else "")
    return args


def resolve_device(name):
    if name == "auto":
        return get_device()
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("error: --device mps requested but MPS is not available")
    if name not in ("cuda", "mps", "cpu"):
        raise SystemExit(f"error: unknown device {name!r} (use auto, cuda, mps or cpu)")
    return torch.device(name)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.results_dir)
    results_csv = out / "robustness_evaluation.csv"
    pred_dir = out / "robustness_predictions"
    chart_dir = out / "robustness_charts"
    if args.verify:
        return verify(results_csv, pred_dir)
    if args.charts_only:
        return 0 if make_charts(results_csv, chart_dir, args.model) else 1

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}   torch {torch.__version__}")

    try:
        rows = ev.load_test_rows(args.test_csv)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    rows = ev.subsample(rows, args.limit, args.seed)
    fp = ev.rows_fingerprint(rows)
    print(f"test set: {ev.rel(args.test_csv)}  {len(rows):,} images  fingerprint {fp[:12]}"
          + ("  [SMOKE SUBSET]" if args.limit else ""))

    ckpt_path = args.resnet_checkpoint if args.model == "resnet50" else args.clip_checkpoint
    try:
        model, ckpt = ev.LOADERS[args.model](ckpt_path, device)
    except ev.CheckpointError as e:
        sys.exit(f"error: {e}\n(no results written; robustness needs a trained, validated checkpoint)")
    md5 = ev.file_md5(ckpt_path)
    print(f"checkpoint: {ev.rel(ckpt_path)}  md5 {md5[:12]}  epoch {ckpt.get('epoch', '-')}  "
          f"label_map {ckpt['label_map']}")

    ref = issue13_reference(args.model)
    if ref and not args.limit:
        same = ref.get("checkpoint_md5") == md5 and ref.get("test_fingerprint") == fp
        print(f"issue #13 reference: accuracy {ref['accuracy']}  ROC-AUC {ref['roc_auc']}  "
              f"({'same checkpoint + test set' if same else 'DIFFERENT checkpoint or test set'})")

    results = read_results(results_csv)
    sizes = [Image.open(r["path"]).size for r in rows]
    status = "smoke" if args.limit else "evaluated"

    def done(kind, level):
        return next((r for r in results if r["model"] == args.model and r["distortion"] == kind
                     and r["level"] == level_str(kind, level) and r["status"] == status
                     and r["checkpoint_md5"] == md5 and r["test_fingerprint"] == fp), None)

    kinds = list(args.distortions)
    if "clean" not in kinds and (args.force or not done("clean", None)):
        kinds.insert(0, "clean")
    kinds.sort(key=list(DISTORTIONS).index)

    baseline = None
    for kind in kinds:
        for level in DEFAULT_LEVELS[kind]:
            prev = done(kind, level)
            if kind == "clean" and prev and not args.force:
                baseline = prev
            if prev and not args.force:
                print(f"\n== {kind} {level_label(kind, level)}: already done "
                      f"(accuracy {float(prev['accuracy']):.4f}), skipping")
                continue
            if kind != "clean" and baseline is None:
                baseline = done("clean", None)
                if baseline is None:
                    sys.exit("error: clean baseline missing; run with --distortions clean first")
            d = make_distortion(kind, level)
            print(f"\n== {args.model} | {kind} {level_label(kind, level)}")
            t0 = time.time()
            y, yhat, p = predict(model, rows, ev.PREPROCESS[args.model], device, d,
                                 args.batch_size, args.workers)
            m = setting_metrics(y, yhat, p)
            entry = dict.fromkeys(FIELDS, "")
            entry.update(model=args.model, checkpoint=ev.rel(ckpt_path), checkpoint_md5=md5,
                         distortion=kind, level=level_str(kind, level), level_label=level_label(kind, level),
                         status=status, model_selection=MODEL_SELECTION_NOTE[args.model],
                         test_manifest=ev.rel(args.test_csv), test_fingerprint=fp, device=str(device),
                         resample=RESAMPLE_NAME if kind in ("resize", "resize_relative", "crop") else "",
                         elapsed_s=f"{time.time() - t0:.0f}", notes=distortion_notes(kind, level, sizes))
            entry.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()})
            if kind == "clean":
                baseline = entry
                if ref and not args.limit:
                    entry["notes"] = (f"issue #13 reference accuracy {ref['accuracy']}, ROC-AUC {ref['roc_auc']}; "
                                      f"re-measured difference {m['accuracy'] - float(ref['accuracy']):+.6f}")
            ba, br = float(baseline["accuracy"]), float(baseline["roc_auc"])
            drop, drop_pp = accuracy_drop(ba, entry["accuracy"])
            entry.update(baseline_accuracy=ba, accuracy_drop=round(drop, 6), accuracy_drop_pp=round(drop_pp, 4),
                         baseline_roc_auc=br, roc_auc_drop=round(br - entry["roc_auc"], 6))
            write_predictions(pred_path(pred_dir, args.model, kind, level), y, yhat, p)
            results = upsert(results, {k: str(v) for k, v in entry.items()})
            ev.write_csv(results_csv, FIELDS, results)  # saved after every setting
            print(f"  accuracy {m['accuracy']:.4f} (drop {drop_pp:+.2f} pp)   ROC-AUC {m['roc_auc']:.4f}   "
                  f"real {m['real_accuracy']:.4f} / fake {m['fake_accuracy']:.4f}   {time.time() - t0:.0f}s")
            if device.type == "mps":
                torch.mps.empty_cache()

    mine = [r for r in results if r["model"] == args.model]
    print(f"\n{'distortion':16s} {'level':14s} {'n':>6s} {'accuracy':>9s} {'drop pp':>8s} {'ROC-AUC':>8s}")
    for r in mine:
        print(f"{r['distortion']:16s} {r['level_label']:14s} {r['num_images']:>6s} {float(r['accuracy']):9.4f} "
              f"{float(r['accuracy_drop_pp']):+8.2f} {float(r['roc_auc']):8.4f}")
    print(f"\nwrote {ev.rel(results_csv)} and {ev.rel(pred_dir)}/")
    make_charts(results_csv, chart_dir, args.model)
    if args.model == "resnet50":
        print("\nNOTE: ResNet-50 is the provisional model -- the CLIP linear probe has not been "
              "evaluated in-distribution yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
