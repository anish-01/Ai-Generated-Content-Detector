"""Pipeline tests for the ResNet-50 model and training script (issue #9).

    python -m unittest tests.test_resnet50 -v

Needs torch/torchvision and the Issue #2 manifests + images (splits/, data/).
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.dataset import RealFakeDataset, get_device, read_index  # noqa: E402
from models.resnet50 import LABELS, build_resnet50, load_checkpoint, save_checkpoint  # noqa: E402
import train_resnet50  # noqa: E402

def have_data():
    """splits/*.csv are in git, the images aren't -- skip if they're not downloaded."""
    for name in ("train.csv", "val.csv"):
        csv_path = ROOT / "splits" / name
        if not csv_path.exists() or not Path(read_index(csv_path)[0]["path"]).exists():
            return False
    return True


HAVE_DATA = have_data()
SKIP_MSG = "dataset images not downloaded (see README: Data)"


@unittest.skipUnless(HAVE_DATA, SKIP_MSG)
class ModelPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        rows = read_index(ROOT / "splits" / "train.csv")[:8]
        ds = RealFakeDataset(rows, model="resnet", train=True)
        cls.x, cls.y = next(iter(torch.utils.data.DataLoader(ds, batch_size=8)))
        cls.model = build_resnet50().to(cls.device)

    def test_batch_shape_and_normalization(self):
        self.assertEqual(tuple(self.x.shape), (8, 3, 224, 224))
        self.assertTrue(set(self.y.tolist()) <= {0, 1})
        self.assertLess(abs(self.x.mean().item()), 2.0)  # normalized, not 0..255 or 0..1

    def test_two_logit_head(self):
        self.assertEqual(self.model.fc.out_features, 2)
        self.assertEqual(LABELS, {0: "real", 1: "fake"})
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.x.to(self.device))
        self.assertEqual(tuple(out.shape), (8, 2))

    def test_loss_backward_and_step(self):
        model = build_resnet50().to(self.device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.fc.weight.detach().clone()
        loss = nn.CrossEntropyLoss()(model(self.x.to(self.device)), self.y.to(self.device))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.fc.weight.grad)
        opt.step()
        self.assertFalse(torch.equal(before, model.fc.weight.detach()))

    def test_checkpoint_roundtrip_across_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ck.pth"
            opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
            save_checkpoint(path, self.model, opt, None, epoch=3, best_val_loss=0.5)
            restored, ckpt = load_checkpoint(path, device="cpu")
            self.assertEqual(ckpt["epoch"], 3)
            self.assertEqual(ckpt["label_map"], LABELS)
            for k, v in self.model.state_dict().items():
                self.assertTrue(torch.equal(v.cpu(), restored.state_dict()[k]), k)
            self.assertEqual(next(restored.parameters()).device.type, "cpu")


@unittest.skipUnless(HAVE_DATA, SKIP_MSG)
class TrainingSmokeTest(unittest.TestCase):
    """Two tiny epochs, interrupted after one and resumed."""

    def test_train_save_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            common = ["--limit-train", "64", "--limit-val", "32", "--batch-size", "16",
                      "--workers", "0", "--checkpoint-dir", str(tmp / "ck"),
                      "--results-dir", str(tmp / "res"), "--log-every", "2"]
            train_resnet50.main(common + ["--epochs", "1"])
            self.assertTrue((tmp / "ck" / "resnet50_best.pth").exists())
            self.assertTrue((tmp / "ck" / "resnet50_last.pth").exists())

            train_resnet50.main(common + ["--epochs", "2", "--resume"])
            _, ckpt = load_checkpoint(tmp / "ck" / "resnet50_last.pth")
            self.assertEqual(ckpt["epoch"], 2)
            self.assertIsNotNone(ckpt["optimizer_state"])
            self.assertIsNotNone(ckpt["scheduler_state"])
            self.assertEqual(ckpt["config"]["batch_size"], 16)

            with open(tmp / "res" / "resnet50_training_history.csv") as f:
                hist = list(csv.DictReader(f))
            self.assertEqual([int(h["epoch"]) for h in hist], [1, 2])
            for h in hist:
                for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr"):
                    self.assertTrue(0 <= float(h[k]) < 100, (k, h[k]))
            self.assertTrue((tmp / "res" / "resnet50_loss.png").exists())
            self.assertTrue((tmp / "res" / "resnet50_accuracy.png").exists())


if __name__ == "__main__":
    unittest.main()
