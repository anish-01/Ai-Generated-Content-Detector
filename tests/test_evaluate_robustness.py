"""Tests for scripts/evaluate_robustness.py (issue #14).

    python -m unittest tests.test_evaluate_robustness -v

DistortionTest / MetricsTest / OutputTest / PipelineTest use synthetic images and
a tiny stand-in model on CPU; any accuracy they produce only checks the code and
says nothing about real model performance. RealDataTest checks the real test
manifest and the saved full-run results, and is skipped when they are absent.
"""

import csv
import gzip
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_in_distribution as ev  # noqa: E402
import evaluate_robustness as rb  # noqa: E402

TEST_CSV = ROOT / "splits" / "test.csv"
HOLDOUT_CSV = ROOT / "splits" / "cross_generator_test.csv"
RESULTS = ROOT / "results" / "robustness_evaluation.csv"
PRED_DIR = ROOT / "results" / "robustness_predictions"
MANIFEST_FIELDS = ["image_path", "label", "source_dataset", "generator", "split", "source_id", "md5"]


def noise_image(w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def pixels(img):
    return np.asarray(img, dtype=np.int16)


def make_images(root, n_per_class=6, size=(40, 32)):
    rows = []
    for label, name in ((0, "real"), (1, "fake")):
        for i in range(n_per_class):
            p = root / "imgs" / name / f"{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            v = 20 + 10 * i if label == 0 else 150 + 10 * i
            Image.new("RGB", size, (v, v, v)).save(p)
            rows.append({"image_path": str(p), "label": label, "source_dataset": "synthetic",
                         "generator": "synthetic", "split": "test", "source_id": f"{name}{i}", "md5": ""})
    return rows


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


class BrightnessModel(nn.Module):
    """Stand-in classifier: bright image -> fake. Two logits like the real head."""

    def forward(self, x):
        m = x.mean(dim=(1, 2, 3))
        return torch.stack([-m, m], dim=1) * 5


# ---------------------------------------------------------------- distortions

class DistortionTest(unittest.TestCase):
    def setUp(self):
        self.img = noise_image()

    def test_jpeg_is_real_compression_and_keeps_size(self):
        out = rb.JPEGCompress(10)(self.img)
        self.assertEqual(out.size, self.img.size)
        self.assertEqual(out.mode, "RGB")
        self.assertGreater(np.abs(pixels(out) - pixels(self.img)).mean(), 5)

    def test_jpeg_lower_quality_means_more_error(self):
        err = [np.abs(pixels(rb.JPEGCompress(q)(self.img)) - pixels(self.img)).mean()
               for q in (90, 50, 10)]
        self.assertLess(err[0], err[1])
        self.assertLess(err[1], err[2])

    def test_jpeg_matches_pillow_encoder(self):
        import io
        buf = io.BytesIO()
        self.img.save(buf, format="JPEG", quality=30)
        expected = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        self.assertTrue(np.array_equal(pixels(rb.JPEGCompress(30)(self.img)), pixels(expected)))

    def test_resize_round_trip_restores_original_dimensions(self):
        for s in (192, 128, 64, 32, 8):
            self.assertEqual(rb.ResizeRoundTrip(s)(self.img).size, (64, 48))

    def test_resize_goes_through_target_size(self):
        seen = []
        orig = Image.Image.resize

        def spy(im, size, *a, **k):
            seen.append(tuple(size))
            return orig(im, size, *a, **k)

        with mock.patch.object(Image.Image, "resize", spy):
            rb.ResizeRoundTrip(16)(self.img)
        self.assertEqual(seen, [(16, 16), (64, 48)])

    def test_resize_loses_detail(self):
        out = rb.ResizeRoundTrip(8)(self.img)
        self.assertGreater(np.abs(pixels(out) - pixels(self.img)).mean(), 20)

    def test_relative_resize_scale_and_dims(self):
        d = rb.RelativeResizeRoundTrip(32)
        self.assertAlmostEqual(d.scale, 32 / 224)
        self.assertEqual(d.small_size(32, 32), (5, 5))
        self.assertEqual(d.small_size(512, 384), (73, 55))
        self.assertEqual(d(self.img).size, self.img.size)
        self.assertEqual(rb.RelativeResizeRoundTrip(1).small_size(3, 3), (1, 1))

    def test_crop_box_area_fraction(self):
        for area in (0.9, 0.75, 0.5, 0.25):
            l, t, r, b = rb.crop_box(1000, 800, area)
            self.assertAlmostEqual((r - l) * (b - t) / (1000 * 800), area, delta=0.003)
            # centred
            self.assertLessEqual(abs(l - (1000 - r)), 1)
            self.assertLessEqual(abs(t - (800 - b)), 1)

    def test_crop_box_side_scales_by_sqrt_area(self):
        self.assertEqual(rb.crop_box(32, 32, 0.25), (8, 8, 24, 24))
        self.assertEqual(rb.crop_box(100, 100, 0.5), (14, 14, 85, 85))  # side round(70.71) = 71, left (100-71)//2
        self.assertEqual(rb.crop_box(10, 10, 1.0), (0, 0, 10, 10))

    def test_crop_resized_back_to_original(self):
        for area in (0.9, 0.75, 0.5, 0.25):
            self.assertEqual(rb.CenterCropArea(area)(self.img).size, self.img.size)

    def test_crop_content_is_centre(self):
        img = Image.new("RGB", (40, 40), (0, 0, 0))
        img.paste((255, 255, 255), (10, 10, 30, 30))  # white centre = 25% of area
        out = pixels(rb.CenterCropArea(0.25)(img))
        self.assertGreater(out.mean(), 250)

    def test_original_not_mutated(self):
        before = pixels(self.img).copy()
        for d in (rb.JPEGCompress(10), rb.ResizeRoundTrip(8), rb.RelativeResizeRoundTrip(32),
                  rb.CenterCropArea(0.25)):
            d(self.img)
        self.assertTrue(np.array_equal(before, pixels(self.img)))

    def test_deterministic(self):
        for kind, level in (("jpeg", 30), ("resize", 32), ("resize_relative", 64), ("crop", 0.5)):
            d = rb.make_distortion(kind, level)
            self.assertTrue(np.array_equal(pixels(d(self.img)), pixels(d(self.img))), kind)

    def test_picklable_for_dataloader_workers(self):
        import pickle
        for kind, levels in rb.DEFAULT_LEVELS.items():
            for lv in levels:
                d = pickle.loads(pickle.dumps(rb.make_distortion(kind, lv)))
                self.assertEqual(d(self.img).size, self.img.size)

    def test_identity(self):
        self.assertIs(rb.make_distortion("clean", None)(self.img), self.img)


class ValidationTest(unittest.TestCase):
    def test_bad_jpeg_quality(self):
        for q in (0, 101, -5, 50.5, "50", True):
            with self.assertRaises(ValueError):
                rb.JPEGCompress(q)

    def test_bad_resize(self):
        for s in (0, -1, 3.5, None):
            with self.assertRaises(ValueError):
                rb.ResizeRoundTrip(s)
        with self.assertRaises(ValueError):
            rb.RelativeResizeRoundTrip(300)

    def test_bad_crop(self):
        for a in (0, -0.1, 1.5, "0.5"):
            with self.assertRaises(ValueError):
                rb.CenterCropArea(a)

    def test_unknown_distortion(self):
        with self.assertRaises(ValueError):
            rb.make_distortion("blur", 3)
        with self.assertRaises(ValueError):
            rb.level_label("blur", 3)

    def test_cli_validation(self):
        with self.assertRaises(SystemExit):
            rb.parse_args(["--batch-size", "0"])
        with self.assertRaises(SystemExit):
            rb.parse_args(["--distortions", "blur"])
        with self.assertRaises(SystemExit):
            rb.parse_args(["--limit", "1"])

    def test_smoke_runs_go_to_separate_dir(self):
        self.assertEqual(rb.parse_args(["--limit", "64"]).results_dir.name, "robustness_smoke")
        self.assertEqual(rb.parse_args([]).results_dir, ROOT / "results")

    def test_unavailable_device(self):
        if not torch.cuda.is_available():
            with self.assertRaises(SystemExit):
                rb.resolve_device("cuda")
        with self.assertRaises(SystemExit):
            rb.resolve_device("tpu")
        self.assertEqual(rb.resolve_device("cpu").type, "cpu")


# ---------------------------------------------------------------- metrics

class MetricsTest(unittest.TestCase):
    def test_accuracy_and_counts(self):
        y = np.array([0, 0, 1, 1, 1])
        yhat = np.array([0, 1, 1, 1, 0])
        p = np.array([0.1, 0.6, 0.9, 0.8, 0.4])
        m = rb.setting_metrics(y, yhat, p)
        self.assertEqual((m["num_images"], m["n_correct"]), (5, 3))
        self.assertAlmostEqual(m["accuracy"], 0.6)
        self.assertAlmostEqual(m["real_accuracy"], 0.5)
        self.assertAlmostEqual(m["fake_accuracy"], 2 / 3)

    def test_roc_auc_known_value(self):
        # pairs (fake, real) ranked correctly: 5 of 6 -> 0.8333
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([0.1, 0.6, 0.9, 0.8, 0.4])
        self.assertAlmostEqual(rb.setting_metrics(y, (p >= .5).astype(int), p)["roc_auc"], 5 / 6)

    def test_roc_auc_uses_scores_not_labels(self):
        y = np.array([0, 0, 1, 1])
        yhat = np.array([0, 0, 0, 0])
        p = np.array([0.1, 0.2, 0.3, 0.4])  # all below threshold but perfectly ranked
        m = rb.setting_metrics(y, yhat, p)
        self.assertEqual(m["roc_auc"], 1.0)
        self.assertEqual(m["accuracy"], 0.5)

    def test_accuracy_drop_sign_and_pp(self):
        d, pp = rb.accuracy_drop(0.981486, 0.90)
        self.assertAlmostEqual(d, 0.081486)
        self.assertAlmostEqual(pp, 8.1486)
        d, pp = rb.accuracy_drop("0.95", "0.96")  # improvement keeps its negative sign
        self.assertAlmostEqual(d, -0.01)
        self.assertAlmostEqual(pp, -1.0)
        self.assertEqual(rb.accuracy_drop(0.9, 0.9), (0.0, 0.0))

    def test_label_mapping(self):
        self.assertEqual(ev.EXPECTED_LABEL_MAP, {0: "real", 1: "fake"})
        # p_fake is softmax column 1
        model = BrightnessModel()
        logits = model(torch.full((1, 3, 4, 4), 2.0))
        self.assertGreater(torch.softmax(logits, 1)[0, 1].item(), 0.5)

    def test_metric_errors(self):
        with self.assertRaises(ValueError):
            rb.setting_metrics(np.array([1, 1]), np.array([1, 1]), np.array([.9, .8]))
        with self.assertRaises(ValueError):
            rb.setting_metrics(np.array([0, 1]), np.array([0, 1]), np.array([np.nan, .8]))


# ---------------------------------------------------------------- output files

class OutputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_predictions_round_trip_exactly(self):
        y = np.array([0, 1, 1, 0])
        yhat = np.array([0, 1, 0, 0])
        p = np.array([1e-12, 0.99999994, 0.123456789, 0.5], dtype=np.float32).astype(np.float64)
        path = self.tmp / "p.csv.gz"
        rb.write_predictions(path, y, yhat, p)
        y2, yhat2, p2 = rb.read_predictions(path)
        self.assertTrue(np.array_equal(y, y2) and np.array_equal(yhat, yhat2))
        self.assertTrue(np.array_equal(p.astype(np.float32), p2.astype(np.float32)))
        with gzip.open(path, "rt") as f:
            self.assertEqual(f.readline().strip(), ",".join(rb.PRED_FIELDS))

    def test_results_csv_rejects_foreign_columns(self):
        bad = self.tmp / "r.csv"
        bad.write_text("a,b\n1,2\n")
        with self.assertRaises(ValueError):
            rb.read_results(bad)

    def test_upsert_replaces_and_orders(self):
        def row(kind, level, acc):
            r = dict.fromkeys(rb.FIELDS, "")
            r.update(model="resnet50", distortion=kind, level=level, accuracy=acc)
            return r
        rows = []
        for kind, lv in (("crop", "0.50"), ("jpeg", "10"), ("clean", ""), ("jpeg", "90"), ("crop", "0.90")):
            rows = rb.upsert(rows, row(kind, lv, "0.5"))
        rows = rb.upsert(rows, row("jpeg", "10", "0.7"))
        self.assertEqual([(r["distortion"], r["level"]) for r in rows],
                         [("clean", ""), ("jpeg", "90"), ("jpeg", "10"), ("crop", "0.90"), ("crop", "0.50")])
        self.assertEqual(next(r for r in rows if r["level"] == "10")["accuracy"], "0.7")

    def test_level_strings(self):
        self.assertEqual(rb.level_str("crop", 0.5), "0.50")
        self.assertEqual(rb.level_str("jpeg", 90), "90")
        self.assertEqual(rb.level_str("clean", None), "")
        self.assertEqual(rb.level_label("crop", 0.75), "75% area")


# ---------------------------------------------------------------- end-to-end (synthetic)

class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.manifest = self.tmp / "test.csv"
        write_manifest(self.manifest, make_images(self.tmp))
        self.out = self.tmp / "results"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_main(self, *extra):
        loader = lambda path, device: (BrightnessModel().eval(), {"label_map": {0: "real", 1: "fake"}})
        args = ["--test-csv", str(self.manifest), "--results-dir", str(self.out), "--device", "cpu",
                "--workers", "0", "--batch-size", "5", *extra]
        with mock.patch.dict(ev.LOADERS, {"resnet50": loader}), \
                mock.patch.object(ev, "file_md5", return_value="0" * 32), \
                mock.patch.object(rb, "make_charts", return_value=[]):
            return rb.main(args)

    def test_full_run_csv_format_and_consistency(self):
        self.assertEqual(self.run_main(), 0)
        rows = rb.read_results(self.out / "robustness_evaluation.csv")
        n_settings = sum(len(v) for v in rb.DEFAULT_LEVELS.values())
        self.assertEqual(len(rows), n_settings)
        self.assertEqual(list(rows[0]), rb.FIELDS)
        self.assertEqual(rows[0]["distortion"], "clean")
        self.assertEqual(sum(r["distortion"] == "clean" for r in rows), 1)
        base = float(rows[0]["accuracy"])
        for r in rows:
            self.assertEqual(r["num_images"], "12")
            self.assertEqual(r["status"], "evaluated")
            self.assertTrue(r["model_selection"].startswith("provisional"))
            self.assertAlmostEqual(float(r["baseline_accuracy"]), base)
            self.assertAlmostEqual(float(r["accuracy_drop"]), base - float(r["accuracy"]), places=6)
            self.assertAlmostEqual(float(r["accuracy_drop_pp"]), 100 * float(r["accuracy_drop"]), places=3)
            self.assertAlmostEqual(float(r["accuracy"]), int(r["n_correct"]) / 12, places=6)
        self.assertEqual(rb.verify(self.out / "robustness_evaluation.csv",
                                   self.out / "robustness_predictions"), 0)

    def test_resume_skips_completed_settings(self):
        self.run_main("--distortions", "clean", "jpeg")
        with mock.patch.object(rb, "predict", wraps=rb.predict) as spy:
            self.run_main("--distortions", "clean", "jpeg", "crop")
        self.assertEqual(spy.call_count, len(rb.DEFAULT_LEVELS["crop"]))  # only the new ones
        rows = rb.read_results(self.out / "robustness_evaluation.csv")
        self.assertEqual(len(rows), 1 + 5 + 4)

    def test_clean_run_automatically_first(self):
        self.run_main("--distortions", "crop")
        rows = rb.read_results(self.out / "robustness_evaluation.csv")
        self.assertEqual(rows[0]["distortion"], "clean")

    def test_reproducible(self):
        self.run_main("--distortions", "jpeg")
        first = (self.out / "robustness_evaluation.csv").read_text()
        self.run_main("--distortions", "jpeg", "--force")
        strip = lambda t: [l.rsplit(",", 2)[0] for l in t.splitlines()]  # ignore elapsed_s, notes
        self.assertEqual(strip(first), strip((self.out / "robustness_evaluation.csv").read_text()))

    def test_verify_detects_tampering(self):
        self.run_main("--distortions", "clean")
        path = self.out / "robustness_evaluation.csv"
        rows = rb.read_results(path)
        rows[0]["accuracy"] = "0.1"
        ev.write_csv(path, rb.FIELDS, rows)
        self.assertEqual(rb.verify(path, self.out / "robustness_predictions"), 1)

    def test_missing_checkpoint_writes_nothing(self):
        def missing(path, device):
            raise ev.CheckpointError("checkpoint not found")
        with mock.patch.dict(ev.LOADERS, {"resnet50": missing}):
            with self.assertRaises(SystemExit) as cm:
                rb.main(["--test-csv", str(self.manifest), "--results-dir", str(self.out),
                         "--device", "cpu", "--workers", "0"])
        self.assertIn("checkpoint not found", str(cm.exception.code))
        self.assertFalse((self.out / "robustness_evaluation.csv").exists())

    def test_missing_manifest_error(self):
        with self.assertRaises(SystemExit) as cm:
            rb.main(["--test-csv", str(self.tmp / "nope.csv"), "--device", "cpu"])
        self.assertIn("not found", str(cm.exception.code))

    def test_holdout_manifest_refused(self):
        holdout = self.tmp / "cross_generator_test.csv"
        shutil.copy(self.manifest, holdout)
        with self.assertRaises(SystemExit) as cm:
            rb.main(["--test-csv", str(holdout), "--device", "cpu"])
        self.assertIn("holdout", str(cm.exception.code))


# ---------------------------------------------------------------- real data

@unittest.skipUnless(TEST_CSV.exists() and HOLDOUT_CSV.exists(), "split manifests not present")
class RealDataTest(unittest.TestCase):
    def test_holdout_excluded_from_test_set(self):
        with open(TEST_CSV) as f:
            test = list(csv.DictReader(f))
        with open(HOLDOUT_CSV) as f:
            hold = list(csv.DictReader(f))
        self.assertFalse({r["image_path"] for r in test} & {r["image_path"] for r in hold})
        self.assertFalse({r["md5"] for r in test} & {r["md5"] for r in hold})
        self.assertEqual({r["split"] for r in test}, {"test"})

    @unittest.skipUnless(RESULTS.exists(), "full robustness run not present")
    def test_saved_results(self):
        rows = [r for r in rb.read_results(RESULTS) if r["status"] == "evaluated"]
        clean = [r for r in rows if r["distortion"] == "clean"]
        self.assertEqual(len(clean), 1)
        for r in rows:
            self.assertEqual(r["num_images"], "18148")
            self.assertEqual(r["test_fingerprint"], clean[0]["test_fingerprint"])
            self.assertEqual(r["checkpoint_md5"], clean[0]["checkpoint_md5"])
        # re-measured baseline agrees with issue #13 (same checkpoint + images)
        self.assertAlmostEqual(float(clean[0]["accuracy"]), 0.981486, delta=0.001)
        self.assertEqual(rb.verify(RESULTS, PRED_DIR), 0)


if __name__ == "__main__":
    unittest.main()
