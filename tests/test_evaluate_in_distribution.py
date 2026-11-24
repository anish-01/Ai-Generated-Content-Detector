"""Tests for scripts/evaluate_in_distribution.py (issue #13).

    python -m unittest tests.test_evaluate_in_distribution -v

MetricsTest / CheckpointTest / ManifestTest / PipelineTest use synthetic data
and small stand-in models; their numbers only check the code and say nothing
about real model performance. RealDataTest checks the real test manifest,
the real ResNet-50 checkpoint and the saved results, and is skipped for any of
those that are not present locally.
"""

import csv
import importlib.util
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

_spec = importlib.util.spec_from_file_location("evaluate_in_distribution",
                                               ROOT / "scripts" / "evaluate_in_distribution.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

TEST_CSV = ROOT / "splits" / "test.csv"
HOLDOUT_CSV = ROOT / "splits" / "cross_generator_test.csv"
RESNET_CKPT = ROOT / "checkpoints" / "resnet50_best.pth"
RESULTS = ROOT / "results" / "in_distribution_evaluation.csv"
PREDICTIONS = ROOT / "results" / "in_distribution_predictions.csv"
MANIFEST_FIELDS = ["image_path", "label", "source_dataset", "generator", "split", "source_id", "md5"]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def make_images(root, n_per_class=6):
    """Tiny solid-colour PNGs: real ones dark, fake ones bright."""
    rows = []
    for label, name in ((0, "real"), (1, "fake")):
        for i in range(n_per_class):
            p = root / "imgs" / name / f"{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            v = 20 + 10 * i if label == 0 else 140 + 10 * i
            Image.new("RGB", (32, 32), (v, v, v)).save(p)
            rows.append({"image_path": str(p), "label": label, "source_dataset": "synthetic",
                         "generator": "synthetic", "split": "test", "source_id": f"{name}{i}", "md5": ""})
    return rows


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


class BrightnessModel(nn.Module):
    """Stand-in model: logit for 'fake' grows with mean pixel value."""

    def forward(self, x):
        s = x.mean(dim=(1, 2, 3)) * 4
        return torch.stack([torch.zeros_like(s), s], dim=1)


class ThreeLogitModel(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 3)


# ---------------------------------------------------------------- metrics

class MetricsTest(unittest.TestCase):
    def test_accuracy(self):
        m = ev.compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.4])
        self.assertAlmostEqual(m["accuracy"], 0.5)

    def test_roc_auc_known_value(self):
        # pairs (real, fake): (0.1,.35)+ (0.1,.8)+ (0.4,.35)- (0.4,.8)+ -> 3/4
        m = ev.compute_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertAlmostEqual(m["roc_auc"], 0.75)

    def test_roc_auc_uses_continuous_scores(self):
        # every score is above 0.5, so hard labels are all "fake" (acc 0.5, AUC 0.5
        # if computed from labels) but the ranking is perfect
        y, p = [0, 0, 1, 1], [0.6, 0.7, 0.8, 0.9]
        m = ev.compute_metrics(y, p)
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["roc_auc"], 1.0)

    def test_roc_curve_returned(self):
        fpr, tpr, _ = ev.compute_metrics([0, 1, 0, 1], [0.2, 0.9, 0.3, 0.6])["roc"]
        self.assertEqual((fpr[0], tpr[0], fpr[-1], tpr[-1]), (0, 0, 1, 1))

    def test_explicit_predictions_used_for_accuracy(self):
        m = ev.compute_metrics([0, 1], [0.2, 0.9], y_pred=[1, 1])
        self.assertAlmostEqual(m["accuracy"], 0.5)

    def test_requires_both_classes(self):
        with self.assertRaisesRegex(ValueError, "both REAL"):
            ev.compute_metrics([1, 1, 1], [0.2, 0.5, 0.9])

    def test_rejects_non_finite_scores(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            ev.compute_metrics([0, 1], [0.2, float("nan")])

    def test_rejects_scores_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            ev.compute_metrics([0, 1], [-0.1, 2.3])

    def test_rejects_bad_predicted_labels(self):
        with self.assertRaisesRegex(ValueError, "0 \\(real\\) or 1"):
            ev.compute_metrics([0, 1], [0.2, 0.8], y_pred=[0, 2])

    def test_label_mapping(self):
        self.assertEqual(ev.EXPECTED_LABEL_MAP, {0: "real", 1: "fake"})
        self.assertEqual(ev.LABELS, {0: "real", 1: "fake"})


# ---------------------------------------------------------------- checkpoints

class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def save(self, obj, name="ck.pth"):
        p = self.tmp / name
        torch.save(obj, p)
        return p

    def resnet_meta(self, **over):
        meta = {"model_state": {"fc.weight": torch.zeros(2, 2048), "fc.bias": torch.zeros(2)},
                "architecture": "resnet50", "num_classes": 2, "label_map": {0: "real", 1: "fake"}}
        meta.update(over)
        return meta

    def test_missing_resnet_checkpoint(self):
        with self.assertRaisesRegex(ev.CheckpointError, "not found"):
            ev.load_resnet50(self.tmp / "nope.pth")

    def test_missing_clip_checkpoint(self):
        with self.assertRaisesRegex(ev.CheckpointError, "not found"):
            ev.load_clip(self.tmp / "clip_linear_best.pth")

    def test_directory_is_not_a_checkpoint(self):
        with self.assertRaisesRegex(ev.CheckpointError, "directory"):
            ev.load_resnet50(self.tmp)

    def test_wrong_architecture(self):
        with self.assertRaisesRegex(ev.CheckpointError, "architecture"):
            ev.load_resnet50(self.save(self.resnet_meta(architecture="vgg16")))

    def test_wrong_label_map(self):
        with self.assertRaisesRegex(ev.CheckpointError, "label_map"):
            ev.load_resnet50(self.save(self.resnet_meta(label_map={0: "fake", 1: "real"})))

    def test_head_must_have_two_outputs(self):
        state = {"fc.weight": torch.zeros(3, 2048), "fc.bias": torch.zeros(3)}
        with self.assertRaisesRegex(ev.CheckpointError, "expected \\(2, 2048\\)"):
            ev.load_resnet50(self.save(self.resnet_meta(model_state=state)))

    def test_incomplete_weights_rejected(self):
        # right head shape but no backbone weights -> strict load fails
        with self.assertRaisesRegex(ev.CheckpointError, "don't match"):
            ev.load_resnet50(self.save(self.resnet_meta()))

    def test_resnet_checkpoint_rejected_as_clip(self):
        with self.assertRaisesRegex(ev.CheckpointError, "head_state"):
            ev.load_clip(self.save(self.resnet_meta()))

    def test_clip_head_shape_checked(self):
        ck = {"head_state": {"weight": torch.zeros(3, 768), "bias": torch.zeros(3)},
              "architecture": "clip_linear", "embed_dim": 768, "arch": "ViT-L-14",
              "pretrained": "openai", "label_map": {0: "real", 1: "fake"}}
        with self.assertRaisesRegex(ev.CheckpointError, "head weight shape"):
            ev.load_clip(self.save(ck))


# ---------------------------------------------------------------- manifest

class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.rows = make_images(self.tmp, 2)

    def test_cross_generator_holdout_refused(self):
        p = write_manifest(self.tmp / "cross_generator_test.csv", self.rows)
        with self.assertRaisesRegex(ValueError, "cross-generator"):
            ev.load_test_rows(p)

    def test_non_test_rows_refused(self):
        rows = [dict(r) for r in self.rows]
        rows[0]["split"] = "cross_generator_test"
        with self.assertRaisesRegex(ValueError, "non-test"):
            ev.load_test_rows(write_manifest(self.tmp / "t.csv", rows))

    def test_invalid_label_refused(self):
        rows = [dict(r) for r in self.rows]
        rows[0]["label"] = 2
        with self.assertRaisesRegex(ValueError, "label"):
            ev.load_test_rows(write_manifest(self.tmp / "t.csv", rows))

    def test_missing_image_reported(self):
        rows = self.rows + [{**self.rows[0], "image_path": str(self.tmp / "gone.png")}]
        with self.assertRaisesRegex(FileNotFoundError, "1 of 5 test images are missing"):
            ev.load_test_rows(write_manifest(self.tmp / "t.csv", rows))

    def test_missing_manifest_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "test manifest not found"):
            ev.load_test_rows(self.tmp / "absent.csv")

    def test_subsample_keeps_both_classes(self):
        rows = ev.load_test_rows(write_manifest(self.tmp / "t.csv", self.rows))
        sub = ev.subsample(rows, 2, seed=0)
        self.assertEqual(sorted(r["label"] for r in sub), [0, 1])


# ---------------------------------------------------------------- inference + full pipeline

class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.manifest = write_manifest(self.tmp / "test.csv", make_images(self.tmp, 6))
        self.rows = ev.load_test_rows(self.manifest)

    def test_predict_counts_order_and_softmax(self):
        y, yhat, p = ev.predict(BrightnessModel(), self.rows, "resnet", "cpu",
                                batch_size=5, workers=0, log_every=0)
        self.assertEqual(len(p), len(self.rows))
        self.assertEqual(y.tolist(), [r["label"] for r in self.rows])
        self.assertTrue(np.all((p > 0) & (p < 1)))
        np.testing.assert_array_equal(yhat, (p > 0.5).astype(int))
        self.assertGreater(len(np.unique(p)), 2)  # continuous, not just 0/1

    def test_predict_rejects_wrong_output_shape(self):
        with self.assertRaisesRegex(RuntimeError, "output shape"):
            ev.predict(ThreeLogitModel(), self.rows, "resnet", "cpu", workers=0, log_every=0)

    def run_main(self, loaders):
        out = self.tmp / "results"
        with mock.patch.dict(ev.LOADERS, loaders):
            code = ev.main(["--test-csv", str(self.manifest), "--results-dir", str(out),
                            "--workers", "0", "--batch-size", "4", "--device", "cpu"])
        return code, out

    def test_both_models_same_images_and_csv_matches_metrics(self):
        fake = lambda path, device: (BrightnessModel(), {"epoch": 1, "label_map": {0: "real", 1: "fake"}})
        (self.tmp / "ck.pth").write_bytes(b"x")
        with mock.patch.object(ev, "file_md5", return_value="0" * 32):
            code, out = self.run_main({"resnet50": fake, "clip_linear": fake})
        self.assertEqual(code, 0)
        summary = {r["model"]: r for r in read(out / "in_distribution_evaluation.csv")}
        preds = read(out / "in_distribution_predictions.csv")
        by_model = {m: [r for r in preds if r["model"] == m] for m in ev.MODELS}
        for m, rs in by_model.items():
            self.assertEqual(len(rs), len(self.rows))
            self.assertEqual(int(summary[m]["test_images"]), len(self.rows))
            y = [int(r["label"]) for r in rs]
            p = [float(r["p_fake"]) for r in rs]
            yhat = [int(r["pred_label"]) for r in rs]
            m_ = ev.compute_metrics(y, p, yhat)
            self.assertAlmostEqual(float(summary[m]["accuracy"]), m_["accuracy"], places=6)
            self.assertAlmostEqual(float(summary[m]["roc_auc"]), m_["roc_auc"], places=6)
        self.assertEqual([r["image_path"] for r in by_model["resnet50"]],
                         [r["image_path"] for r in by_model["clip_linear"]])
        self.assertEqual(summary["resnet50"]["test_fingerprint"], summary["clip_linear"]["test_fingerprint"])

    def test_missing_model_recorded_as_blocked_without_metrics(self):
        fake = lambda path, device: (BrightnessModel(), {"epoch": 1, "label_map": {0: "real", 1: "fake"}})

        def missing(path, device):
            raise ev.CheckpointError("CLIP linear checkpoint not found: x")

        with mock.patch.object(ev, "file_md5", return_value="0" * 32):
            code, out = self.run_main({"resnet50": fake, "clip_linear": missing})
        self.assertEqual(code, 0)
        summary = {r["model"]: r for r in read(out / "in_distribution_evaluation.csv")}
        self.assertEqual(summary["clip_linear"]["status"], "blocked")
        self.assertEqual((summary["clip_linear"]["accuracy"], summary["clip_linear"]["roc_auc"],
                          summary["clip_linear"]["test_images"]), ("", "", ""))
        models = {r["model"] for r in read(out / "in_distribution_predictions.csv")}
        self.assertEqual(models, {"resnet50"})

    def test_all_models_missing_exits_nonzero(self):
        def missing(path, device):
            raise ev.CheckpointError("not found")

        code, _ = self.run_main({"resnet50": missing, "clip_linear": missing})
        self.assertEqual(code, 1)


# ---------------------------------------------------------------- real data

class RealDataTest(unittest.TestCase):
    @unittest.skipUnless(TEST_CSV.exists(), "splits/test.csv not generated")
    def test_real_manifest(self):
        rows = ev.load_test_rows(TEST_CSV, check_files=False)
        labels = [r["label"] for r in rows]
        self.assertEqual(len(rows), 18148)
        self.assertEqual((labels.count(0), labels.count(1)), (9074, 9074))
        if HOLDOUT_CSV.exists():
            holdout = {r["image_path"] for r in read(HOLDOUT_CSV)}
            self.assertFalse(holdout & {ev.rel(r["path"]) for r in rows})
            self.assertFalse(any(r["generator"] == "midjourney" for r in rows))

    @unittest.skipUnless(HOLDOUT_CSV.exists(), "cross_generator_test.csv not generated")
    def test_real_holdout_refused(self):
        with self.assertRaises(ValueError):
            ev.load_test_rows(HOLDOUT_CSV, check_files=False)

    @unittest.skipUnless(RESNET_CKPT.exists(), "ResNet-50 checkpoint not available locally")
    def test_real_resnet_checkpoint(self):
        model, ckpt = ev.load_resnet50(RESNET_CKPT)
        self.assertFalse(model.training)
        self.assertEqual(model.fc.out_features, 2)
        self.assertEqual(ckpt["label_map"], {0: "real", 1: "fake"})
        with torch.inference_mode():
            out = model(torch.zeros(3, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (3, 2))

    @unittest.skipUnless(RESULTS.exists() and PREDICTIONS.exists(), "evaluation not run yet")
    def test_saved_results_match_predictions(self):
        summary = read(RESULTS)
        preds = read(PREDICTIONS)
        test_paths = [r["image_path"] for r in read(TEST_CSV)]
        for s in summary:
            rs = [r for r in preds if r["model"] == s["model"]]
            if s["status"] != "evaluated":
                self.assertEqual((s["accuracy"], s["roc_auc"]), ("", ""))
                self.assertFalse(rs, f"{s['model']} is {s['status']} but has predictions")
                continue
            self.assertEqual(int(s["test_images"]), len(test_paths))
            self.assertEqual(sorted(r["image_path"] for r in rs), sorted(test_paths))
            m = ev.compute_metrics([int(r["label"]) for r in rs], [float(r["p_fake"]) for r in rs],
                                   [int(r["pred_label"]) for r in rs])
            self.assertAlmostEqual(float(s["accuracy"]), m["accuracy"], places=5)
            self.assertAlmostEqual(float(s["roc_auc"]), m["roc_auc"], places=5)


if __name__ == "__main__":
    unittest.main()
