"""Download a small per-generator subset of GenImage (issue #3).

Pulls from the nebula/GenImage-arrow mirror on Hugging Face, which stores the
original GenImage test set split by generator (6k real + 6k fake each). We only
download the shards we actually need for N real + N fake per generator, dump
the raw image bytes to disk, then delete the shard.

Output layout:
    data/genimage/<generator>/real/*.JPEG
    data/genimage/<generator>/fake/*.png
    data/genimage/index.csv      (path,label,generator,source_name)

Usage:
    python scripts/download_genimage.py                    # sd14, midjourney, biggan, 3000/class
    python scripts/download_genimage.py --per-class 1000   # smaller
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "nebula/GenImage-arrow"

# short folder name -> generator dir name in the mirror
GENERATORS = {
    "sd14": "stable_diffusion_v_1_4",
    "midjourney": "Midjourney",
    "biggan": "BigGAN",
    "sd15": "stable_diffusion_v_1_5",
    "adm": "ADM",
    "glide": "glide",
    "vqdm": "VQDM",
    "wukong": "wukong",
}

ROOT = Path(__file__).resolve().parents[1]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def shards_needed(bundle, per_class):
    """Rows are stored real-first then fake, so figure out which shards
    cover rows [0, n) and [n_real, n_real + n)."""
    n_real = bundle["counts"]["real"]
    want = [(0, min(per_class, n_real)),
            (n_real, n_real + min(per_class, bundle["counts"]["fake"]))]
    picked, start = [], 0
    for shard in bundle["shards"]:
        end = start + shard["rows"]
        if any(start < hi and end > lo for lo, hi in want):
            picked.append(shard)
        start = end
    return picked


def read_rows(path):
    with pa.memory_map(str(path)) as src:
        reader = pa.ipc.open_stream(src)
        for batch in reader:
            d = batch.to_pydict()
            yield from zip(d["image_path"], d["image"], d["label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", nargs="+", default=["sd14", "midjourney", "biggan"],
                    choices=list(GENERATORS))
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "genimage")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / ".hf_cache")
    ap.add_argument("--keep-shards", action="store_true", help="don't delete .arrow files after extracting")
    args = ap.parse_args()

    manifest = json.load(open(hf_hub_download(REPO, "manifest.json", repo_type="dataset",
                                              cache_dir=args.cache)))
    rows_out = []

    for short in args.generators:
        bundle = manifest["bundles"]["test"][GENERATORS[short]]
        shards = shards_needed(bundle, args.per_class)
        counts = {0: 0, 1: 0}
        print(f"\n[{short}] {len(shards)} shard(s), "
              f"{sum(s['size_bytes'] for s in shards) / 1e9:.2f} GB to download")

        for shard in shards:
            local = Path(hf_hub_download(REPO, shard["path"], repo_type="dataset",
                                         cache_dir=args.cache))
            if sha256(local) != shard["sha256"]:
                raise RuntimeError(f"checksum mismatch on {shard['path']}")

            for img_path, img_bytes, label in tqdm(read_rows(local), total=shard["rows"],
                                                   desc=Path(shard["path"]).name, leave=False):
                if counts[label] >= args.per_class:
                    continue
                cls = "fake" if label == 1 else "real"
                name = os.path.basename(img_path)
                dest = args.out / short / cls / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(img_bytes)
                counts[label] += 1
                rel = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
                rows_out.append((rel.as_posix(), label, short, name))

            if not args.keep_shards:
                # hf cache stores blobs behind symlinks; remove the real file too
                real = local.resolve()
                local.unlink(missing_ok=True)
                real.unlink(missing_ok=True)

        print(f"[{short}] real={counts[0]} fake={counts[1]}")
        if min(counts.values()) < args.per_class:
            print(f"[{short}] warning: got fewer than {args.per_class} for a class")

    args.out.mkdir(parents=True, exist_ok=True)
    index = args.out / "index.csv"
    with open(index, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "generator", "source_name"])
        w.writerows(rows_out)

    # GenImage reuses ImageNet val photos as the "real" half for every generator,
    # so the same real photo can show up under more than one generator folder.
    # Flag it here so the train/val/test split (#2) can dedupe on source_name.
    real_names = [r[3] for r in rows_out if r[1] == 0]
    dupes = len(real_names) - len(set(real_names))
    print(f"\nwrote {len(rows_out)} rows to {index}")
    print(f"real images shared across generators: {dupes}")


if __name__ == "__main__":
    main()
