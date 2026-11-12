"""Check the CIFAKE dataset (issue #1).

CIFAKE is 120k 32x32 images: CIFAR-10 photos (REAL) and Stable Diffusion v1.4
images (FAKE), split into train/test.
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

To save disk space this never extracts the images. It downloads the ~110 MB
zip to a temp file, reads everything straight from the zip, and deletes it
when done (pass --zip PATH to reuse a zip you already have).

It:
  - counts REAL (0) / FAKE (1) per split and checks balance
  - opens every image to catch corrupt / odd-sized files
  - looks for exact duplicates across classes (possible label conflicts)
  - picks 20 REAL + 20 FAKE (10 train + 10 test each, fixed seed) and saves
    labeled contact sheets to docs/cifake_samples/
  - writes the results to docs/cifake_audit.md

The dataset is public, so no Kaggle login is needed.

Usage:
    python scripts/check_cifake.py
    python scripts/check_cifake.py --seed 7
    python scripts/check_cifake.py --zip ~/Downloads/archive.zip
"""

import argparse
import hashlib
import io
import random
import sys
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageStat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Same label rules as models/dataset.py (0 = real, 1 = fake). That module
# imports torch, so fall back to an identical copy if torch isn't installed.
try:
    from models.dataset import IMG_EXTS, label_from_path
except ImportError:
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def label_from_path(path):
        for part in reversed(PurePosixPath(path).parts[:-1]):
            if part.lower() in {"real", "nature"}:
                return 0
            if part.lower() in {"fake", "ai"}:
                return 1
        return None

URL = "https://www.kaggle.com/api/v1/datasets/download/birdy654/cifake-real-and-ai-generated-synthetic-images"
PAGE = "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"
SPLITS = ("train", "test")
CLASSES = {0: "REAL", 1: "FAKE"}
NOTES_MARKER = "<!-- manual notes below this line are kept when the script reruns -->"


def download(dest):
    print(f"downloading CIFAKE zip (~110 MB) to temp file {dest}")
    req = urllib.request.Request(URL, headers={"User-Agent": "cifake-check"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def scan(z):
    rows = []
    names = [n for n in z.namelist() if PurePosixPath(n).suffix.lower() in IMG_EXTS]
    for i, name in enumerate(names, 1):
        p = PurePosixPath(name)
        split = next((s for s in p.parts if s.lower() in SPLITS), None)
        label = label_from_path(name)
        if split is None or label is None:
            continue
        data = z.read(name)
        row = {"name": name, "file": p.name, "split": split.lower(), "label": label,
               "md5": hashlib.md5(data).hexdigest(), "error": ""}
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                row["size"], row["mode"] = im.size, im.mode
        except Exception as e:
            row.update(error=f"{type(e).__name__}: {e}", size=None, mode=None)
        rows.append(row)
        if i % 10000 == 0:
            print(f"\r  checked {i:,}/{len(names):,}", end="", flush=True)
    print(f"\r  checked {len(names):,}/{len(names):,}")
    return rows


def count_table(rows):
    c = defaultdict(int)
    for r in rows:
        c[(r["split"], r["label"])] += 1
    table = [("split", "REAL (0)", "FAKE (1)", "total", "balanced")]
    for s in SPLITS + ("all",):
        real = sum(v for (sp, l), v in c.items() if l == 0 and s in (sp, "all"))
        fake = sum(v for (sp, l), v in c.items() if l == 1 and s in (sp, "all"))
        table.append((s, f"{real:,}", f"{fake:,}", f"{real + fake:,}",
                      "yes" if real == fake else f"no ({real - fake:+,})"))
    return table


def find_issues(rows):
    sizes, modes, by_hash = defaultdict(int), defaultdict(int), defaultdict(list)
    for r in rows:
        if not r["error"]:
            sizes[r["size"]] += 1
            modes[r["mode"]] += 1
        by_hash[r["md5"]].append(r)
    groups = [g for g in by_hash.values() if len(g) > 1]
    return {
        "corrupt": [r for r in rows if r["error"]],
        "odd_size": [r for r in rows if not r["error"] and r["size"] != (32, 32)],
        "sizes": sizes, "modes": modes, "dup_groups": groups,
        "cross_class": [g for g in groups if len({r["label"] for r in g}) > 1],
        "cross_split": [g for g in groups if len({r["split"] for r in g}) > 1],
    }


def pick_samples(rows, seed, per_split=10):
    rng = random.Random(seed)
    picks = []
    for label in (0, 1):
        for split in SPLITS:
            pool = sorted((r for r in rows if r["label"] == label and r["split"] == split
                           and not r["error"]), key=lambda r: r["name"])
            picks += rng.sample(pool, min(per_split, len(pool)))
    return picks


def flags_for(im):
    """Cheap 'take a look' hints -- they don't prove a label is wrong."""
    stat = ImageStat.Stat(im.convert("RGB"))
    mean, std = sum(stat.mean) / 3, sum(stat.stddev) / 3
    flags = []
    if std < 12:
        flags.append("near-flat")
    if mean < 20 or mean > 235:
        flags.append("very dark/bright")
    return round(mean, 1), round(std, 1), flags


def contact_sheet(images, title, out, cols=5, scale=4):
    cell, pad, cap = 32 * scale, 8, 30
    nrows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + pad) + pad, 36 + nrows * (cell + cap + pad) + pad), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 12), title, fill="black")
    for i, (r, im) in enumerate(images):
        x, y = pad + (i % cols) * (cell + pad), 36 + (i // cols) * (cell + cap + pad)
        sheet.paste(im.convert("RGB").resize((cell, cell), Image.NEAREST), (x, y))
        d.text((x, y + cell + 3), f"#{i + 1} {r['split']}", fill="black")
        d.text((x, y + cell + 15), r["file"][:20], fill="dimgray")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def md_table(lines):
    out = ["| " + " | ".join(map(str, lines[0])) + " |", "|" + "---|" * len(lines[0])]
    return "\n".join(out + ["| " + " | ".join(map(str, l)) + " |" for l in lines[1:]])


def write_report(path, table, issues, samples, seed, sheets):
    notes = "## Manual review notes\n\n_Add notes from looking at the contact sheets here._"
    if path.exists() and NOTES_MARKER in path.read_text():
        notes = path.read_text().split(NOTES_MARKER, 1)[1].strip() or notes

    sample_lines = [("#", "class", "split", "file", "opens", "mean", "std", "auto flags")]
    for cls in (0, 1):
        for i, (r, mean, std, flags) in enumerate((s for s in samples if s[0]["label"] == cls), 1):
            sample_lines.append((i, CLASSES[cls], r["split"], f"`{r['file']}`", "yes",
                                 mean, std, ", ".join(flags) or "-"))

    corrupt = "".join(f"\n  - `{r['name']}`: {r['error']}" for r in issues["corrupt"][:20])
    dupes = "".join("\n  - " + ", ".join(f"`{r['name']}`" for r in g)
                      for g in issues["cross_class"][:10])
    rel = lambda p: p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)

    path.write_text(f"""# CIFAKE dataset audit (issue #1)

Generated by `python scripts/check_cifake.py --seed {seed}` from the [Kaggle CIFAKE zip]({PAGE}),
read directly from the archive (images are not stored in the repo). Labels follow
`models/dataset.py`: REAL = 0, FAKE = 1.

## Class distribution (counted from the files)

{md_table(table)}

## Integrity

- Failed to open: **{len(issues['corrupt'])}**{corrupt}
- Sizes: {", ".join(f"{w}x{h}: {n:,}" for (w, h), n in sorted(issues['sizes'].items(), key=lambda kv: -kv[1]))}
- Colour modes: {", ".join(f"{m}: {n:,}" for m, n in issues['modes'].items())}
- Not 32x32: **{len(issues['odd_size'])}**
- Exact-duplicate groups (identical bytes): **{len(issues['dup_groups'])}**
  - in both REAL and FAKE (possible label conflict): **{len(issues['cross_class'])}**{dupes}
  - in both train and test (leakage): **{len(issues['cross_split'])}**

## Visual sample (seed {seed})

20 REAL + 20 FAKE, 10 train + 10 test per class. Contact sheets:
{chr(10).join(f"- `{rel(p)}`" for p in sheets)}

Auto flags only mean "worth a look". At 32x32 you can't prove an image is real or
AI-generated by eye, so doubtful ones are marked for manual review, not relabelled.

{md_table(sample_lines)}

{NOTES_MARKER}

{notes}
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", type=Path, help="use an existing CIFAKE zip instead of downloading")
    ap.add_argument("--report", type=Path, default=ROOT / "docs" / "cifake_audit.md")
    ap.add_argument("--sheets", type=Path, default=ROOT / "docs" / "cifake_samples")
    args = ap.parse_args()

    tmp = None
    zpath = args.zip
    if zpath is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        zpath = Path(tmp.name)
    try:
        if tmp:
            download(zpath)
        with zipfile.ZipFile(zpath) as z:
            print("checking images")
            rows = scan(z)
            if not rows:
                sys.exit("no train/test REAL/FAKE images found in the zip")
            table, issues = count_table(rows), find_issues(rows)
            picks = pick_samples(rows, args.seed)
            loaded = [(r, Image.open(io.BytesIO(z.read(r["name"])))) for r in picks]
    finally:
        if tmp:
            zpath.unlink(missing_ok=True)
            print("deleted temp zip")

    widths = [max(len(str(l[i])) for l in table) for i in range(len(table[0]))]
    print("\nCIFAKE class distribution")
    for j, l in enumerate(table):
        print("  " + "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                               for i, (c, w) in enumerate(zip(l, widths))))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))
    print(f"\nfailed to open: {len(issues['corrupt'])}   not 32x32: {len(issues['odd_size'])}   "
          f"duplicate groups: {len(issues['dup_groups'])} "
          f"(REAL+FAKE: {len(issues['cross_class'])}, train+test: {len(issues['cross_split'])})")

    samples = [(r, *flags_for(im)) for r, im in loaded]
    sheets = []
    for cls in (0, 1):
        out = args.sheets / f"{CLASSES[cls].lower()}_seed{args.seed}.png"
        contact_sheet([x for x in loaded if x[0]["label"] == cls],
                      f"CIFAKE {CLASSES[cls]} (label {cls}) - seed {args.seed}", out)
        sheets.append(out)

    print(f"\nsamples (seed {args.seed}):")
    for r, mean, std, flags in samples:
        print(f"  {CLASSES[r['label']]:4s} {r['split']:5s} {r['file']:22s} "
              f"mean {mean:5.1f} std {std:5.1f}  {', '.join(flags) or 'ok'}")

    write_report(args.report, table, issues, samples, args.seed, sheets)
    print(f"\ncontact sheets: {', '.join(str(p) for p in sheets)}\nreport: {args.report}")


if __name__ == "__main__":
    main()
