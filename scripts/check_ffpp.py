"""Check the local FaceForensics++ download before building on it.

Video detection in this project samples frames (~1 fps), scores each with the
image model and pools the scores (mean / top-k). So for each video we need:
  - it decodes with OpenCV (same reader extract_ffpp_frames.py uses)
  - it's long enough to give at least MIN_SECONDS frames at 1 fps
  - the short side is at least 224 px (model input size)
And for the dataset as a whole:
  - every method folder is complete (1000 videos in FF++)
  - every fake's target video (000 in 000_003) exists in the real folder,
    so we can split by source video without leaks
  - real vs fake counts are known so we can balance

This only reads the videos, it never modifies them. Results go to
docs/ffpp_audit.md, which is safe to commit (no frames or video in it; FF++
itself can't be redistributed).

Usage:
    python scripts/check_ffpp.py --ffpp-root C:/Users/you/FaceForensics
    python scripts/check_ffpp.py --ffpp-root ... --full-decode   # slow, counts every frame
"""

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
EXPECTED_PER_FOLDER = 1000  # FF++ has 1000 originals and 1000 videos per method
MIN_SECONDS = 5             # enough 1 fps frames for top-k pooling (k <= 5)
MIN_SHORT_SIDE = 224        # model input size


def probe(path, full_decode=False):
    """Return basic stats for one video. Never raises."""
    info = {"path": str(path), "size_mb": path.stat().st_size / 1e6, "ok": False,
            "error": "", "fps": 0.0, "frames": 0, "width": 0, "height": 0, "codec": ""}
    if path.stat().st_size == 0:
        info["error"] = "empty file"
        return info
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            info["error"] = "OpenCV could not open it"
            return info
        info["fps"] = cap.get(cv2.CAP_PROP_FPS) or 0.0
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        info["codec"] = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if full_decode:
            n = 0
            while cap.grab():
                n += 1
            if n == 0:
                info["error"] = "no decodable frames"
                return info
        else:
            # first frame + one near the end catches empty and truncated files
            ok, _ = cap.read()
            if not ok:
                info["error"] = "first frame unreadable"
                return info
            if n > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(n - 2, 0))
                ok, _ = cap.read()
                if not ok:
                    info["error"] = f"frame {n - 2} unreadable (truncated?)"
                    return info
        info["frames"] = n
        info["ok"] = True
        return info
    finally:
        cap.release()


def _probe_star(args):
    return probe(*args)


def collect(ffpp_root, compression):
    """Return {(label, group): [paths]} and stray non-video files."""
    groups, strays = {}, []
    layout = []
    orig = ffpp_root / "original_sequences"
    if orig.is_dir():
        layout += [(0, p.name, p / compression / "videos") for p in sorted(orig.iterdir()) if p.is_dir()]
    manip = ffpp_root / "manipulated_sequences"
    if manip.is_dir():
        layout += [(1, p.name, p / compression / "videos") for p in sorted(manip.iterdir()) if p.is_dir()]
    for label, name, vdir in layout:
        vids = []
        if vdir.is_dir():
            for f in sorted(vdir.iterdir()):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                    vids.append(f)
                elif f.is_file():
                    strays.append(f)
        groups[(label, name)] = vids
    return groups, strays


def summarize(vals):
    if not vals:
        return "-"
    return f"{min(vals):.1f} / {statistics.median(vals):.1f} / {statistics.mean(vals):.1f} / {max(vals):.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ffpp-root", required=True, type=Path)
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--full-decode", action="store_true", help="decode every frame (exact counts, slow)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "ffpp_audit.md")
    args = ap.parse_args()

    root = args.ffpp_root.expanduser()
    if not root.is_dir():
        sys.exit(f"not a folder: {root}")
    groups, strays = collect(root, args.compression)
    if not groups:
        sys.exit(f"no original_sequences/ or manipulated_sequences/ under {root}")

    jobs = [(p, args.full_decode) for vids in groups.values() for p in vids]
    print(f"probing {len(jobs)} videos with {args.workers} workers...")
    with Pool(args.workers) as pool:
        results = {}
        for i, r in enumerate(pool.imap_unordered(_probe_star, jobs, chunksize=8), 1):
            results[r["path"]] = r
            if i % 250 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")

    real_ids = {p.stem for (label, _), vids in groups.items() if label == 0 for p in vids}

    rows, problems, short, small = [], [], [], []
    all_ok = []
    missing_target = defaultdict(list)
    for (label, name), vids in groups.items():
        rs = [results[str(p)] for p in vids]
        ok = [r for r in rs if r["ok"]]
        all_ok += ok
        for r in rs:
            if not r["ok"]:
                problems.append((name, Path(r["path"]).name, r["error"]))
        secs = [r["frames"] / r["fps"] for r in ok if r["fps"] > 0]
        for r in ok:
            dur = r["frames"] / r["fps"] if r["fps"] > 0 else 0
            if dur < MIN_SECONDS:
                short.append((name, Path(r["path"]).name, dur))
            if min(r["width"], r["height"]) < MIN_SHORT_SIDE:
                small.append((name, Path(r["path"]).name, f'{r["width"]}x{r["height"]}'))
        if label == 1:
            for p in vids:
                target = p.stem.split("_")[0]
                if target not in real_ids:
                    missing_target[name].append(p.stem)
        rows.append({
            "label": "real" if label == 0 else "fake", "name": name, "n": len(vids), "ok": len(ok),
            "secs": secs, "frames": [r["frames"] for r in ok],
            "gb": sum(r["size_mb"] for r in rs) / 1000,
            "samples_1fps": sum(int(s) + 1 for s in secs),
        })

    codecs = Counter(r["codec"] for r in all_ok)
    fpss = Counter(round(r["fps"], 2) for r in all_ok)
    res = Counter(f'{r["width"]}x{r["height"]}' for r in all_ok)

    incomplete = [r for r in rows if r["n"] < EXPECTED_PER_FOLDER]
    complete_methods = [r["name"] for r in rows if r["label"] == "fake" and r["n"] >= EXPECTED_PER_FOLDER]
    n_real = sum(r["ok"] for r in rows if r["label"] == "real")
    n_fake = sum(r["ok"] for r in rows if r["label"] == "fake")

    # ---- verdict checklist
    checks = [
        ("All videos decode in OpenCV", not problems, f"{len(problems)} failed"),
        (f"All clips >= {MIN_SECONDS}s (enough 1 fps frames for top-k)", not short, f"{len(short)} too short"),
        (f"Short side >= {MIN_SHORT_SIDE}px", not small, f"{len(small)} too small"),
        (f"Every folder has {EXPECTED_PER_FOLDER} videos", not incomplete,
         ", ".join(f'{r["name"]} {r["n"]}' for r in incomplete)),
        ("Every fake's target real video is present", not missing_target,
         f"{sum(len(v) for v in missing_target.values())} fakes missing their real video"),
        (">= 2 complete methods (one to train, one held out)", len(complete_methods) >= 2,
         ", ".join(complete_methods) or "none"),
        ("No leftover temp/partial files", not strays, f"{len(strays)} found"),
    ]

    L = []
    L.append("# FaceForensics++ audit\n")
    L.append(f"Generated by `scripts/check_ffpp.py` on {date.today().isoformat()}. "
             f"Compression: `{args.compression}`. "
             f"{'Exact frame counts (full decode).' if args.full_decode else 'Frame counts from the container header; first and last frames test-decoded.'}\n")
    L.append("FF++ is gated: you need to submit the request form and accept the terms of use to get the "
             "download script, and the videos can't be redistributed. Keep them out of the repo "
             "(`data/` is gitignored).\n")
    L.append("## Verdict\n")
    L.append("| Check | Result | Detail |\n|---|---|---|")
    for name, passed, detail in checks:
        L.append(f"| {name} | {'PASS' if passed else '**FAIL**'} | {'' if passed else detail} |")
    L.append("")

    L.append("## Per folder\n")
    L.append("Duration and frames are min / median / mean / max.\n")
    L.append("| Label | Folder | Videos | Decodes | Duration (s) | Frames | ~Frames at 1 fps | Size (GB) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f'| {r["label"]} | {r["name"]} | {r["n"]} / {EXPECTED_PER_FOLDER} | {r["ok"]} | '
                 f'{summarize(r["secs"])} | {summarize(r["frames"])} | {r["samples_1fps"]:,} | {r["gb"]:.2f} |')
    L.append(f"\nUsable videos: {n_real} real, {n_fake} fake.\n")

    L.append("## Format\n")
    L.append(f"- Codec (FourCC): {', '.join(f'{k or chr(63)} ({v})' for k, v in codecs.most_common())}")
    L.append(f"- FPS: {', '.join(f'{k} ({v})' for k, v in sorted(fpss.items()))}")
    L.append(f"- Resolutions ({len(res)} distinct, top 8): "
             + ", ".join(f"{k} ({v})" for k, v in res.most_common(8)))
    L.append("")

    if problems or strays or short or small or missing_target:
        L.append("## Issues\n")
        for name, f, err in problems:
            L.append(f"- Unreadable: `{name}/{f}`: {err}")
        for f in strays:
            L.append(f"- Non-video file: `{f.relative_to(root).as_posix()}` ({f.stat().st_size} bytes). "
                     "Probably left by an interrupted download, so re-run the download for that folder.")
        for name, f, d in short:
            L.append(f"- Short: `{name}/{f}` ({d:.1f}s)")
        for name, f, wh in small:
            L.append(f"- Small: `{name}/{f}` ({wh})")
        for name, ids in missing_target.items():
            L.append(f"- {name}: {len(ids)} fakes have no matching real video (e.g. {', '.join(ids[:5])})")
        L.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
