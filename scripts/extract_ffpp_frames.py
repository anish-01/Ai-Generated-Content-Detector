"""Pull frames out of FaceForensics++ videos at ~1 fps.

Expects the layout the official FF++ download script produces:
    <ffpp_root>/original_sequences/youtube/<comp>/videos/000.mp4
    <ffpp_root>/original_sequences/actors/<comp>/videos/...          (optional, DFD)
    <ffpp_root>/manipulated_sequences/<Method>/<comp>/videos/000_003.mp4

Output layout:
    data/ffpp/real/<source>/<video>/<video>_f0000.png
    data/ffpp/fake/<Method>/<video>/<video>_f0025.png
    data/ffpp/index.csv   (path,label,generator,video,source_video,frame,compression)

Frame numbers in filenames are the frame's position in the original video, so
reruns with the same settings give the same files. Splits should be done by
video (or better, by source_video) and never by frame. Frames from one clip are
near-duplicates, and fakes like 000_003 reuse the background of real video 000.

Videos that already have a .done marker get skipped, so an interrupted run can
just be started again.

Usage:
    # dry run on a few videos first
    python scripts/extract_ffpp_frames.py --ffpp-root D:/FaceForensics --limit 1 --out data/ffpp_test
    # full run
    python scripts/extract_ffpp_frames.py --ffpp-root D:/FaceForensics --workers 8
"""

import argparse
import csv
import json
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
FALLBACK_FPS = 25.0  # most FF++ clips are 25-30 fps


def find_videos(ffpp_root, compression, methods, limit):
    """Yield dicts describing each video to process."""
    jobs = []
    groups = []  # (label, generator, videos_dir)
    orig = ffpp_root / "original_sequences"
    if orig.is_dir():
        for src in sorted(p for p in orig.iterdir() if p.is_dir()):
            groups.append((0, src.name, src / compression / "videos"))
    manip = ffpp_root / "manipulated_sequences"
    if manip.is_dir():
        for m in sorted(p for p in manip.iterdir() if p.is_dir()):
            if methods and m.name not in methods:
                continue
            groups.append((1, m.name, m / compression / "videos"))

    for label, gen, vdir in groups:
        if not vdir.is_dir():
            print(f"  skip {gen}: no {vdir.relative_to(ffpp_root)}")
            continue
        vids = sorted(p for p in vdir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if limit:
            vids = vids[:limit]
        print(f"  {gen:16s} {'fake' if label else 'real'}  {len(vids)} videos")
        for v in vids:
            jobs.append({"video": str(v), "label": label, "generator": gen})
    return jobs


def out_dir_for(job, out_root):
    cls = "fake" if job["label"] else "real"
    return out_root / cls / job["generator"] / Path(job["video"]).stem


def extract(args):
    """Worker: decode one video, keep ~target_fps frames. Returns (job, frames, info)."""
    job, out_root, target_fps, max_frames, ext, jpeg_q = args
    vpath = Path(job["video"])
    odir = out_dir_for(job, out_root)
    done = odir / ".done"
    if done.exists():
        info = json.loads(done.read_text())
        return job, info["frames"], {**info, "skipped": True}

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return job, [], {"error": "could not open"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:  # 0, NaN, or garbage
        fps = FALLBACK_FPS
    step = max(1, round(fps / target_fps))

    if odir.exists():  # partial output from an interrupted run
        shutil.rmtree(odir)
    odir.mkdir(parents=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_q] if ext == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]

    # read sequentially with grab()/retrieve(); seeking with CAP_PROP_POS_FRAMES
    # is not frame-accurate on h264 and can silently return the wrong frame
    frames, idx, size = [], 0, None
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, img = cap.retrieve()
            if not ok or img is None:
                break
            name = f"{vpath.stem}_f{idx:04d}.{ext}"
            if not cv2.imwrite(str(odir / name), img, params):
                cap.release()
                return job, [], {"error": f"imwrite failed for {name}"}
            frames.append([name, idx])
            size = img.shape[1], img.shape[0]
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if not frames:
        return job, [], {"error": "no frames decoded"}
    info = {"fps": fps, "step": step, "decoded": idx, "size": size, "frames": frames}
    done.write_text(json.dumps(info))
    return job, frames, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ffpp-root", type=Path, required=True, help="folder with original_sequences/ and manipulated_sequences/")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ffpp")
    ap.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    ap.add_argument("--methods", nargs="+", help="only these manipulation methods (default: all found)")
    ap.add_argument("--fps", type=float, default=1.0, help="target frames per second (default 1)")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames per video (0 = no cap)")
    ap.add_argument("--format", default="png", choices=["png", "jpg"],
                    help="png is lossless (no extra compression on top of the video's); jpg is ~5-10x smaller")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos per folder (for testing)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    root = args.ffpp_root.resolve()
    out = args.out.resolve()
    print(f"scanning {root} ({args.compression})")
    jobs = find_videos(root, args.compression, set(args.methods or []), args.limit)
    if not jobs:
        raise SystemExit("no videos found, check --ffpp-root and --compression")

    work = [(j, out, args.fps, args.max_frames, args.format, args.jpeg_quality) for j in jobs]
    rows, failed, skipped, sizes = [], [], 0, set()
    t0 = time.time()
    with Pool(args.workers) as pool:
        for n, (job, frames, info) in enumerate(pool.imap_unordered(extract, work), 1):
            if "error" in info:
                failed.append((job["video"], info["error"]))
            else:
                skipped += info.get("skipped", False)
                sizes.add(tuple(info["size"]))
                odir = out_dir_for(job, out)
                stem = Path(job["video"]).stem
                for name, idx in frames:
                    p = odir / name
                    rows.append({
                        "path": (p.relative_to(ROOT) if p.is_relative_to(ROOT) else p).as_posix(),
                        "label": job["label"],
                        "generator": job["generator"],
                        "video": stem,
                        # fakes are named <target>_<source>; the target clip supplies
                        # the background, so group it with real video <target>
                        "source_video": stem.split("_")[0],
                        "frame": idx,
                        "compression": args.compression,
                    })
            if n % 50 == 0 or n == len(work):
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(work)} videos  {len(rows)} frames  "
                      f"{rate:.1f} vid/s  eta {(len(work) - n) / rate / 60:.1f} min", flush=True)

    rows.sort(key=lambda r: (r["label"], r["generator"], r["video"], r["frame"]))
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.csv"
    with open(index, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)

    # summary
    print(f"\nwrote {len(rows)} frames from {len(jobs) - len(failed)} videos to {index}")
    if skipped:
        print(f"  ({skipped} videos already done, reused)")
    per_gen = {}
    for r in rows:
        per_gen.setdefault(r["generator"], set()).add(r["video"])
    for g, vids in sorted(per_gen.items()):
        n = sum(1 for r in rows if r["generator"] == g)
        print(f"  {g:16s} {len(vids):5d} videos  {n:6d} frames  ({n / len(vids):.1f}/video)")
    print(f"  frame sizes seen: {sorted(sizes)[:5]}{' ...' if len(sizes) > 5 else ''}")
    if rows:
        total_bytes = sum((ROOT / r["path"] if not Path(r["path"]).is_absolute() else Path(r["path"])).stat().st_size
                          for r in rows)
        print(f"  disk: {total_bytes / 1e6:.1f} MB, {total_bytes / len(rows) / 1e3:.0f} KB/frame")
    if failed:
        print(f"\n{len(failed)} videos failed:")
        for v, e in failed:
            print(f"  {v}: {e}")


if __name__ == "__main__":
    main()
