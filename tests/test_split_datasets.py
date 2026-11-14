"""Tests for scripts/split_datasets.py (issue #2).

    python -m unittest tests.test_split_datasets -v

SyntheticSplitTest builds a tiny fake CIFAKE + GenImage tree with known
leakage traps and checks the script handles them. RealSplitTest re-checks the
real manifests in splits/ and loads them with RealFakeDataset; it is skipped
if the manifests haven't been generated.
"""

import csv
import importlib
import random
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "split_datasets.py"
PY = sys.executable


def noise_image(path, seed, size=32, fmt=None):
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size))
    img.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(size * size)])
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format=fmt)


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run_split(tmp, *extra):
    cmd = [PY, str(SCRIPT), "--cifake", str(tmp / "cifake"),
           "--genimage-index", str(tmp / "genimage" / "index.csv"),
           "--out", str(tmp / "splits"), "--report", str(tmp / "report.md"),
           "--tolerance", "0.05", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def make_synthetic(tmp):
    """CIFAKE: 40 per (split, class). GenImage: gens 'sd14', 'biggan',
    'midjourney' with 40 real + 40 fake each. Traps:
      - midjourney real 'shared_0' also appears under sd14 (re-encoded bytes,
        same source_name)  -> sd14 copy must be excluded
      - a CIFAKE file identical to a midjourney fake -> must be excluded
      - an identical image under CIFAKE REAL in both train and test folders
        -> must stay in one split
      - one corrupt CIFAKE file -> excluded as unreadable
    """
    seed = 0
    for split in ("train", "test"):
        for cls in ("REAL", "FAKE"):
            for i in range(40):
                seed += 1
                noise_image(tmp / "cifake" / split / cls / f"{i:03d}.jpg", seed, fmt="PNG")
    dup = (tmp / "cifake" / "train" / "REAL" / "000.jpg").read_bytes()
    (tmp / "cifake" / "test" / "REAL" / "dup_of_train.jpg").write_bytes(dup)
    (tmp / "cifake" / "test" / "FAKE" / "broken.jpg").write_bytes(b"not an image")

    rows = []
    for gen in ("sd14", "biggan", "midjourney"):
        for label, cls in ((0, "real"), (1, "fake")):
            for i in range(40):
                seed += 1
                name = f"{gen}_{cls}_{i:03d}.png"
                p = tmp / "genimage" / gen / cls / name
                noise_image(p, seed, fmt="PNG")
                rows.append((str(p), label, gen, name))
    # shared real: same ImageNet source_name, different bytes
    shared = tmp / "genimage" / "midjourney" / "real" / "shared_0.JPEG"
    noise_image(shared, 999, fmt="PNG")
    rows.append((str(shared), 0, "midjourney", "shared_0.JPEG"))
    copy = tmp / "genimage" / "sd14" / "real" / "shared_0.JPEG"
    noise_image(copy, 999, fmt="BMP")
    rows.append((str(copy), 0, "sd14", "shared_0.JPEG"))
    # CIFAKE file identical to a midjourney fake
    mj_fake = tmp / "genimage" / "midjourney" / "fake" / "midjourney_fake_000.png"
    (tmp / "cifake" / "train" / "FAKE" / "same_as_mj.jpg").write_bytes(mj_fake.read_bytes())

    with open(tmp / "genimage" / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "generator", "source_name"])
        w.writerows(rows)


class SyntheticSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        make_synthetic(cls.tmp)
        cls.result = run_split(cls.tmp)
        cls.splits = {s: read(cls.tmp / "splits" / f"{s}.csv")
                      for s in ("train", "val", "test", "cross_generator_test", "excluded")}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def normal(self):
        return [r for s in ("train", "val", "test") for r in self.splits[s]]

    def test_script_succeeds(self):
        self.assertEqual(self.result.returncode, 0, self.result.stdout + self.result.stderr)
        self.assertIn("all checks passed", self.result.stdout)

    def test_required_columns(self):
        for rows in self.splits.values():
            for col in ("image_path", "label", "source_dataset", "generator"):
                self.assertIn(col, rows[0])

    def test_holdout_only_in_cross_generator_test(self):
        self.assertFalse([r for r in self.normal() if r["generator"] == "midjourney"])
        cross = self.splits["cross_generator_test"]
        self.assertEqual({r["generator"] for r in cross}, {"midjourney"})
        self.assertEqual(len(cross), 81)  # 40 real + 40 fake + shared_0

    def test_shared_real_copy_excluded(self):
        names = [Path(r["image_path"]).name for r in self.normal()]
        self.assertNotIn("shared_0.JPEG", names)
        excluded = {Path(r["image_path"]).name: r["reason"] for r in self.splits["excluded"]}
        self.assertIn("held-out", excluded["shared_0.JPEG"])

    def test_cifake_copy_of_holdout_excluded(self):
        excluded = {Path(r["image_path"]).name: r["reason"] for r in self.splits["excluded"]}
        self.assertIn("held-out", excluded["same_as_mj.jpg"])

    def test_corrupt_file_excluded(self):
        excluded = {Path(r["image_path"]).name: r["reason"] for r in self.splits["excluded"]}
        self.assertIn("unreadable", excluded["broken.jpg"])

    def test_identical_files_share_a_split(self):
        where = {"/".join(Path(r["image_path"]).parts[-3:]): r["split"]
                 for r in self.normal() if r["source_dataset"] == "cifake"}
        self.assertEqual(where["train/REAL/000.jpg"], where["test/REAL/dup_of_train.jpg"])

    def test_no_overlap_between_splits(self):
        splits_per_hash = {}
        for r in self.normal() + self.splits["cross_generator_test"]:
            splits_per_hash.setdefault(r["md5"], set()).add(r["split"])
        leaked = {h: s for h, s in splits_per_hash.items() if len(s) > 1}
        self.assertFalse(leaked)

    def test_ratios_and_strata(self):
        n = len(self.normal())
        for s, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            self.assertAlmostEqual(len(self.splits[s]) / n, target, delta=0.05)
            gens = {(r["source_dataset"], r["generator"], r["label"]) for r in self.splits[s]}
            self.assertEqual(len(gens), 6, f"{s} is missing a stratum: {gens}")

    def test_reproducible(self):
        before = {s: (self.tmp / "splits" / f"{s}.csv").read_bytes()
                  for s in ("train", "val", "test", "cross_generator_test")}
        again = run_split(self.tmp)
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        for s, data in before.items():
            self.assertEqual((self.tmp / "splits" / f"{s}.csv").read_bytes(), data, s)

    def test_different_seed_changes_split(self):
        before = (self.tmp / "splits" / "train.csv").read_bytes()
        other = run_split(self.tmp, "--seed", "7", "--out", str(self.tmp / "splits7"))
        self.assertEqual(other.returncode, 0, other.stdout + other.stderr)
        self.assertNotEqual((self.tmp / "splits7" / "train.csv").read_bytes(), before)

    def test_missing_genimage_index_fails_cleanly(self):
        r = subprocess.run([PY, str(SCRIPT), "--cifake", str(self.tmp / "cifake"),
                            "--genimage-index", str(self.tmp / "nope.csv"),
                            "--out", str(self.tmp / "x"), "--report", str(self.tmp / "x.md")],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("GenImage index not found", r.stderr)


def import_loader():
    """models.dataset needs torch/torchvision. If they aren't installed, stub
    the few names it imports so the CSV-reading and filtering logic can still
    be exercised (tensor transforms are replaced with an identity transform)."""
    try:
        import torch  # noqa: F401
        stubbed = False
    except ImportError:
        torch = types.ModuleType("torch")
        data = types.ModuleType("torch.utils.data")
        data.Dataset, data.DataLoader = object, object
        torch.utils = types.ModuleType("torch.utils")
        torch.utils.data = data
        tv = types.ModuleType("torchvision")
        tv.transforms = types.ModuleType("torchvision.transforms")
        sys.modules.update({"torch": torch, "torch.utils": torch.utils, "torch.utils.data": data,
                            "torchvision": tv, "torchvision.transforms": tv.transforms})
        stubbed = True
    sys.path.insert(0, str(ROOT))
    return importlib.import_module("models.dataset"), stubbed


@unittest.skipUnless((ROOT / "splits" / "train.csv").exists(), "splits/ not generated yet")
class RealSplitTest(unittest.TestCase):
    def test_verify_only_passes(self):
        r = subprocess.run([PY, str(SCRIPT), "--verify-only"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_loader_reads_manifests(self):
        dataset, stubbed = import_loader()
        for split in ("train", "val", "test", "cross_generator_test"):
            csv_path = ROOT / "splits" / f"{split}.csv"
            n = len(read(csv_path))
            ds = dataset.RealFakeDataset(str(csv_path), split=split, transform=lambda im: im)
            self.assertEqual(len(ds), n)
            img, label = ds[0]
            self.assertEqual(img.mode, "RGB")
            self.assertIn(label, (0, 1))
        if stubbed:
            print("\n  (torch not installed: loader tested with torch stubbed, identity transform)")


if __name__ == "__main__":
    unittest.main()
