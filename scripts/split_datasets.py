"""Split CIFAKE + GenImage into train/val/test (70/15/15) with one GenImage
generator held out for the cross-generator test (issue #2).

Inputs (produced by the data scripts, see README):
    data/cifake/{train,test}/{REAL,FAKE}/*.jpg
    data/genimage/index.csv          (path,label,generator,source_name)

Outputs (CSV manifests only, images are never copied or moved):
    splits/train.csv, val.csv, test.csv
    splits/cross_generator_test.csv  every image of the held-out generator
    splits/excluded.csv              images left out of every split, with a reason
    docs/dataset_split_report.md

Leakage handling:
  - every image is hashed (MD5 of the file bytes); identical files form one group
  - GenImage "real" images are ImageNet val photos that can appear under several
    generators, so reals with the same source_name are grouped too
  - a whole group always lands in a single split
  - any group touching the held-out generator is removed from the normal splits
    (the held-out copies go to cross_generator_test.csv, the rest to excluded.csv)
  - near-duplicates (resized / re-encoded copies) are NOT detected

The split is stratified by (source_dataset, generator, label) and is fully
determined by --seed.

Usage:
    python scripts/split_datasets.py                 # build + verify + write report
    python scripts/split_datasets.py --verify-only   # re-check existing manifests
"""

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Same label rules as models/dataset.py (0 = real, 1 = fake). That module
# imports torch at the top, so fall back to an identical copy without it.
try:
    from models.dataset import IMG_EXTS, label_from_path
except ImportError:
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def label_from_path(path):
        for part in reversed(Path(path).parts[:-1]):
            if part.lower() in {"real", "nature"}:
                return 0
            if part.lower() in {"fake", "ai"}:
                return 1
        return None

SPLITS = ("train", "val", "test")
HOLDOUT_SPLIT = "cross_generator_test"
COLUMNS = ["image_path", "label", "source_dataset", "generator", "split", "source_id", "md5"]
HOLDOUT_PREFERENCE = ["midjourney"]


# ---------------------------------------------------------------- loading

def rel_path(p):
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def abs_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_cifake(root):
    """CIFAKE keeps its own train/test folders; we pool them and re-split."""
    root = Path(root)
    if not root.is_dir():
        sys.exit(f"CIFAKE not found at {root} (see README: CIFAKE)")
    rows, skipped = [], []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(root)
        label = label_from_path(rel)
        if label is None:
            skipped.append((rel_path(p), "cifake", "cifake", "", "no REAL/FAKE folder in path"))
            continue
        rows.append({"image_path": rel_path(p), "label": label, "source_dataset": "cifake",
                     "generator": "cifake", "source_id": "cifake/" + rel.as_posix()})
    if not rows:
        sys.exit(f"no labelled images under {root}")
    return rows, skipped


def load_genimage(index_csv):
    index_csv = Path(index_csv)
    if not index_csv.exists():
        sys.exit(f"GenImage index not found at {index_csv} "
                 "(run: python scripts/download_genimage.py)")
    rows, skipped = [], []
    with open(index_csv) as f:
        for r in csv.DictReader(f):
            path = rel_path(abs_path(r["path"]))
            label = int(r["label"])
            folder_label = label_from_path(r["path"])
            if folder_label != label:
                skipped.append((path, "genimage", r["generator"], r["source_name"],
                                f"label {label} doesn't match folder ({folder_label})"))
                continue
            rows.append({"image_path": path, "label": label, "source_dataset": "genimage",
                         "generator": r["generator"], "source_id": r["source_name"]})
    if not rows:
        sys.exit(f"{index_csv} has no usable rows")
    return rows, skipped


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def check_files(rows):
    """Hash every file and make sure it opens. Returns (good_rows, bad_rows)."""
    good, bad = [], []
    for i, r in enumerate(rows, 1):
        p = abs_path(r["image_path"])
        try:
            r["md5"] = file_md5(p)
            with Image.open(p) as im:
                im.verify()
            good.append(r)
        except FileNotFoundError:
            bad.append((r, "missing file"))
        except Exception as e:
            bad.append((r, f"unreadable: {type(e).__name__}"))
        if i % 20000 == 0:
            print(f"\r  checked {i:,}/{len(rows):,}", end="", flush=True)
    print(f"\r  checked {len(rows):,}/{len(rows):,}")
    return good, bad


# ---------------------------------------------------------------- grouping

def group_keys(r):
    keys = ["md5:" + r["md5"]]
    if r["source_dataset"] == "genimage" and r["label"] == 0:
        keys.append("imagenet:" + r["source_id"])  # same real photo under several generators
    return keys


def build_groups(rows):
    """Union-find over shared keys -> list of groups (lists of rows)."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_seen = {}
    for i, r in enumerate(rows):
        for k in group_keys(r):
            if k in first_seen:
                parent[find(i)] = find(first_seen[k])
            else:
                first_seen[k] = i
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r)
    return sorted(groups.values(), key=lambda g: min(r["image_path"] for r in g))


def pick_holdout(generators, requested):
    if requested:
        if requested not in generators:
            sys.exit(f"held-out generator {requested!r} not in GenImage data: {sorted(generators)}")
        return requested, "requested on the command line"
    for g in HOLDOUT_PREFERENCE:
        if g in generators:
            return g, "Midjourney is available and is the preferred holdout"
    g = sorted(generators)[0]
    return g, f"Midjourney not available; picked {g} (first alphabetically)"


def separate_holdout(groups, holdout):
    """Groups touching the held-out generator leave the normal pool entirely."""
    pool, holdout_rows, excluded = [], [], []
    for g in groups:
        if any(r["generator"] == holdout for r in g):
            for r in g:
                if r["generator"] == holdout:
                    holdout_rows.append(r)
                else:
                    excluded.append((r, f"duplicate/shared with held-out generator {holdout}"))
        else:
            pool.append(g)
    return pool, holdout_rows, excluded


# ---------------------------------------------------------------- splitting

def stratum(group):
    """A group is filed under its most common (dataset, generator, label)."""
    c = Counter((r["source_dataset"], r["generator"], r["label"]) for r in group)
    return min(c, key=lambda k: (-c[k], str(k)))


def assign_splits(groups, ratios, seed):
    """Within each stratum, shuffle groups (seeded) and give each one to the
    split furthest below its target count. Keeps ratios tight even with
    multi-image groups."""
    by_stratum = defaultdict(list)
    for g in groups:
        by_stratum[stratum(g)].append(g)

    out = []
    for key in sorted(by_stratum, key=str):
        gs = by_stratum[key]
        random.Random(f"{seed}-{key}").shuffle(gs)
        total = sum(len(g) for g in gs)
        filled = dict.fromkeys(SPLITS, 0)
        for g in gs:
            split = max(SPLITS, key=lambda s: (ratios[s] * total - filled[s], -SPLITS.index(s)))
            filled[split] += len(g)
            out += [{**r, "split": split} for r in g]
    return out


# ---------------------------------------------------------------- writing

def write_csv(path, rows, columns=COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image_path"]))


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


# ---------------------------------------------------------------- verification

def verify(out_dir, holdout, ratios, tol=0.01):
    """Re-read the manifests from disk and check every invariant.
    Returns a list of (check, passed, detail)."""
    m = {s: read_csv(out_dir / f"{s}.csv") for s in SPLITS + (HOLDOUT_SPLIT,)}
    normal = [r for s in SPLITS for r in m[s]]
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    seen = defaultdict(set)
    for s, rows in m.items():
        for r in rows:
            seen[r["image_path"]].add(s)
    check("no image path in more than one manifest", all(len(v) == 1 for v in seen.values()))

    md5_splits = defaultdict(set)
    for s, rows in m.items():
        for r in rows:
            md5_splits[r["md5"]].add(s)
    multi = [h for h, v in md5_splits.items() if len(v) > 1]
    check("no identical file content across manifests (MD5)", not multi, f"{len(multi)} hashes shared")

    real_ids = defaultdict(set)
    for s, rows in m.items():
        for r in rows:
            if r["source_dataset"] == "genimage" and r["label"] == 0:
                real_ids[r["source_id"]].add(s)
    shared = [k for k, v in real_ids.items() if len(v) > 1]
    check("no GenImage real source_id across manifests", not shared, f"{len(shared)} shared")

    check(f"held-out generator '{holdout}' absent from train/val/test",
          not any(r["generator"] == holdout for r in normal))
    check("cross_generator_test contains only the held-out generator",
          m[HOLDOUT_SPLIT] and all(r["generator"] == holdout for r in m[HOLDOUT_SPLIT]),
          f"{len(m[HOLDOUT_SPLIT])} images")
    check("split column matches manifest name",
          all(r["split"] == s for s, rows in m.items() for r in rows))

    missing = [r["image_path"] for rows in m.values() for r in rows
               if not abs_path(r["image_path"]).exists()]
    check("every image_path exists", not missing, f"{len(missing)} missing")
    bad_label = [r["image_path"] for rows in m.values() for r in rows
                 if label_from_path(r["image_path"]) != r["label"]]
    check("labels match REAL/FAKE folders", not bad_label, f"{len(bad_label)} mismatched")

    n = len(normal)
    for s in SPLITS:
        frac = len(m[s]) / n if n else 0
        check(f"{s} is {ratios[s]:.0%} +/- {tol:.0%}", abs(frac - ratios[s]) <= tol, f"actual {frac:.2%}")
    for s in SPLITS:
        labels = {r["label"] for r in m[s]}
        check(f"{s} has both REAL and FAKE", labels == {0, 1})
    return results


# ---------------------------------------------------------------- report

def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def class_counts(rows):
    c = Counter(r["label"] for r in rows)
    return c[0], c[1]


def write_report(path, args, inputs, bad, excluded, holdout, reason, out_dir, checks):
    m = {s: read_csv(out_dir / f"{s}.csv") for s in SPLITS + (HOLDOUT_SPLIT,)}
    n_normal = sum(len(m[s]) for s in SPLITS)
    gens = sorted({r["generator"] for r in inputs if r["source_dataset"] == "genimage"})

    ds_rows = []
    for ds in ("cifake", "genimage"):
        rs = [r for r in inputs if r["source_dataset"] == ds]
        real, fake = class_counts(rs)
        ds_rows.append((ds, f"{real:,}", f"{fake:,}", f"{len(rs):,}"))
    gen_rows = []
    for g in gens:
        real, fake = class_counts([r for r in inputs if r["generator"] == g])
        gen_rows.append((g, real, fake, "held out" if g == holdout else "train/val/test"))

    names = {"train": "Training", "val": "Validation", "test": "Testing"}
    split_rows = []
    for s in SPLITS:
        real, fake = class_counts(m[s])
        split_rows.append((names[s], f"{real:,}", f"{fake:,}", f"{len(m[s]):,}",
                           f"{len(m[s]) / n_normal:.2%}"))
    real, fake = class_counts(m[HOLDOUT_SPLIT])
    split_rows.append(("Cross-generator holdout", real, fake, len(m[HOLDOUT_SPLIT]), "separate"))

    strata = sorted({(r["source_dataset"], r["generator"], r["label"])
                     for s in SPLITS for r in m[s]}, key=str)
    strat_rows = []
    for ds, g, lab in strata:
        counts = [sum(1 for r in m[s] if (r["source_dataset"], r["generator"], r["label"]) == (ds, g, lab))
                  for s in SPLITS]
        tot = sum(counts)
        strat_rows.append((ds, g, "REAL" if lab == 0 else "FAKE",
                           *(f"{c:,} ({c / tot:.1%})" for c in counts)))

    reasons = Counter(reason for _, reason in bad) + Counter(r for *_, r in excluded)
    excl_rows = [(k, v) for k, v in sorted(reasons.items())] or [("none", 0)]
    md5_counts = Counter(r["md5"] for r in inputs)
    dup_groups = sum(1 for n in md5_counts.values() if n > 1)
    dup_images = sum(n for n in md5_counts.values() if n > 1)
    check_rows = [("PASS" if ok else "FAIL", name, detail or "-") for name, ok, detail in checks]

    path.write_text(f"""# Dataset split report (issue #2)

Generated by `python scripts/split_datasets.py --seed {args.seed}`. All counts
below are read back from the manifests in `{rel_path(out_dir)}/`.

## Dataset summary

{md_table(("dataset", "REAL (0)", "FAKE (1)", "total"), ds_rows)}

CIFAKE: all images from `{rel_path(args.cifake)}` (its original train/test folders
are pooled and re-split). GenImage: the local subset listed in
`{rel_path(args.genimage_index)}` (downloaded with
`scripts/download_genimage.py --per-class 250`).

GenImage generators available:

{md_table(("generator", "REAL", "FAKE", "use"), gen_rows)}

**Held-out generator: `{holdout}`** ({reason}).

## Split distribution

Ratios are 70/15/15 over the eligible images (after removing the held-out
generator, overlapping copies and unreadable files).

{md_table(("Split", "REAL", "FAKE", "Total", "Percentage"), split_rows)}

Per source / generator / class:

{md_table(("dataset", "generator", "class", "train", "val", "test"), strat_rows)}

## Data integrity

- Images considered: {len(inputs) + len(bad):,}; unreadable or missing: {len(bad)}.
- Byte-identical duplicate groups: {dup_groups:,} ({dup_images:,} images), each kept
  inside a single split.
- Exact duplicates: MD5 of the file bytes. Identical files form one group and
  are always kept in the same split.
- GenImage real images are ImageNet val photos that can be reused across
  generators; reals with the same `source_name` are grouped as well.
- Any group that contains a `{holdout}` image is removed from train/val/test.
  The `{holdout}` images go to `cross_generator_test.csv`; the other copies go
  to `splits/excluded.csv`.
- Stratified by (source_dataset, generator, label), so class balance and each
  generator's share are the same in every split.
- Reproducibility: seed `{args.seed}`, groups sorted before a per-stratum seeded
  shuffle. Rerunning gives byte-identical manifests.

Images left out of every manifest:

{md_table(("reason", "count"), excl_rows)}

Automated checks (re-read from the written CSVs):

{md_table(("result", "check", "detail"), check_rows)}

## Limitations

- **GenImage is a small local subset** (250 REAL + 250 FAKE per generator,
  chosen to fit the available disk space), not the full dataset. CIFAKE
  therefore makes up almost all of train/val/test; weight or resample by
  `source_dataset` when training if GenImage should count for more.
- **Only 3 generators** were downloaded (SD v1.4, BigGAN, Midjourney), so only
  two generators are in the normal splits.
- **Near-duplicate detection is not implemented.** Only byte-identical files and
  known shared ImageNet `source_name`s are caught; resized or re-encoded copies
  would not be.
- CIFAKE's own train/test boundary is not kept: everything is pooled and
  re-split 70/15/15.
- GenImage reals are JPEG and fakes are PNG, so a model could learn the file
  format instead of generator artefacts (see README).
""")


# ---------------------------------------------------------------- main

def build(args):
    ratios = dict(zip(SPLITS, args.ratios))
    cifake, skip_c = load_cifake(args.cifake)
    genimage, skip_g = load_genimage(args.genimage_index)
    print(f"CIFAKE: {len(cifake):,} images   GenImage: {len(genimage):,} images")

    print("hashing and opening every image")
    rows, bad = check_files(cifake + genimage)
    bad += [({"image_path": p, "label": "", "source_dataset": d, "generator": g,
              "source_id": s, "md5": ""}, reason) for p, d, g, s, reason in skip_c + skip_g]

    holdout, reason = pick_holdout({r["generator"] for r in genimage}, args.holdout)
    print(f"held-out generator: {holdout} ({reason})")

    groups = build_groups(rows)
    multi = sum(1 for g in groups if len(g) > 1)
    print(f"{len(rows):,} images in {len(groups):,} groups ({multi:,} groups with >1 image)")

    pool, holdout_rows, excluded = separate_holdout(groups, holdout)
    split_rows = assign_splits(pool, ratios, args.seed)

    out = args.out
    for s in SPLITS:
        write_csv(out / f"{s}.csv", [r for r in split_rows if r["split"] == s])
    write_csv(out / f"{HOLDOUT_SPLIT}.csv", [{**r, "split": HOLDOUT_SPLIT} for r in holdout_rows])
    write_csv(out / "excluded.csv", [{**r, "reason": why} for r, why in bad + excluded],
              COLUMNS[:-3] + ["source_id", "md5", "reason"])
    return rows, bad, excluded, holdout, reason, ratios


def print_checks(checks):
    print("\nverification")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def print_summary(out_dir):
    print(f"\n  {'split':22s} {'REAL':>8s} {'FAKE':>8s} {'total':>8s}")
    for s in SPLITS + (HOLDOUT_SPLIT,):
        real, fake = class_counts(read_csv(out_dir / f"{s}.csv"))
        print(f"  {s:22s} {real:8,} {fake:8,} {real + fake:8,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifake", type=Path, default=ROOT / "data" / "cifake")
    ap.add_argument("--genimage-index", type=Path, default=ROOT / "data" / "genimage" / "index.csv")
    ap.add_argument("--holdout", help="GenImage generator to hold out (default: midjourney if present)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--out", type=Path, default=ROOT / "splits")
    ap.add_argument("--report", type=Path, default=ROOT / "docs" / "dataset_split_report.md")
    ap.add_argument("--tolerance", type=float, default=0.01, help="allowed deviation from each ratio")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if abs(sum(args.ratios) - 1) > 1e-9:
        sys.exit("--ratios must add up to 1")

    if args.verify_only:
        holdout = args.holdout or read_csv(args.out / f"{HOLDOUT_SPLIT}.csv")[0]["generator"]
        checks = verify(args.out, holdout, dict(zip(SPLITS, args.ratios)), args.tolerance)
    else:
        rows, bad, excluded, holdout, reason, ratios = build(args)
        checks = verify(args.out, holdout, ratios, args.tolerance)
        write_report(args.report, args, rows, bad, excluded, holdout, reason, args.out, checks)
        print(f"report: {rel_path(args.report)}")

    print_summary(args.out)
    print_checks(checks)
    if not all(ok for _, ok, _ in checks):
        sys.exit("verification FAILED")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
