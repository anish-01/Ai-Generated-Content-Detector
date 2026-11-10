"""Shared image preprocessing + Dataset for both models (issue #4).

Both ResNet-50 and the CLIP linear probe load data through RealFakeDataset.
The only difference between them is the normalization stats, picked with
get_transform(model="resnet" | "clip").

Labels: 0 = real, 1 = fake (AI-generated).

Quick check:
    python -m models.dataset data/genimage/index.csv
"""

import csv
import io
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = 224

NORM = {
    "resnet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
}

# folder names we treat as real / fake when scanning a directory
REAL_DIRS = {"real", "nature"}
FAKE_DIRS = {"fake", "ai"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------- transforms

def get_transform(model="resnet", train=False):
    """Resize short side to 224, crop to 224x224, to tensor, normalize."""
    mean, std = NORM[model]
    crop = T.RandomCrop(IMG_SIZE) if train else T.CenterCrop(IMG_SIZE)
    steps = [T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC), crop]
    if train:
        steps.append(T.RandomHorizontalFlip())
    steps += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(steps)


# distortions for the robustness tests -- applied to the PIL image *before*
# the normal transform, so they mimic what a social media upload would do

def jpeg(quality):
    def f(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    f.__name__ = f"jpeg_q{quality}"
    return f


def downscale(scale):
    """shrink then scale back up, like a platform resizing an upload"""
    def f(img):
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    f.__name__ = f"resize_{scale}"
    return f


def center_crop(frac):
    def f(img):
        w, h = img.size
        cw, ch = int(w * frac), int(h * frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch))
    f.__name__ = f"crop_{frac}"
    return f


# ---------------------------------------------------------------- indexing

def label_from_path(path):
    for part in reversed(Path(path).parts[:-1]):
        p = part.lower()
        if p in REAL_DIRS:
            return 0
        if p in FAKE_DIRS:
            return 1
    return None


def scan_folder(root, generator=None):
    """Build index rows from a folder tree like <root>/.../{real,fake}/*.jpg.
    Works for both our GenImage layout and CIFAKE's REAL/FAKE folders."""
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        label = label_from_path(p.relative_to(root))
        if label is None:
            continue
        gen = generator or p.relative_to(root).parts[0]
        rows.append({"path": str(p), "label": label, "generator": gen})
    return rows


def read_index(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        if not Path(r["path"]).is_absolute():
            r["path"] = str(ROOT / r["path"])
    return rows


# ---------------------------------------------------------------- dataset

class RealFakeDataset(Dataset):
    """
    source:     index CSV path, folder path, or a list of row dicts
                (each row needs path, label, and optionally generator / split)
    model:      "resnet" or "clip" -- picks normalization
    train:      random crop + flip if True
    generators: only keep these generators (e.g. leave one out for cross-gen test)
    exclude:    drop these generators
    split:      only keep rows whose "split" column matches (if the index has one)
    distortion: optional fn(PIL) -> PIL applied before the transform
    """

    def __init__(self, source, model="resnet", train=False, generators=None,
                 exclude=None, split=None, distortion=None, transform=None):
        if isinstance(source, (list, tuple)):
            rows = list(source)
        elif str(source).endswith(".csv"):
            rows = read_index(source)
        else:
            rows = scan_folder(source)

        if split is not None:
            rows = [r for r in rows if r.get("split") == split]
        if generators is not None:
            rows = [r for r in rows if r.get("generator") in set(generators)]
        if exclude is not None:
            rows = [r for r in rows if r.get("generator") not in set(exclude)]
        if not rows:
            raise ValueError("no images left after filtering")

        self.rows = rows
        self.distortion = distortion
        self.transform = transform or get_transform(model, train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        if self.distortion is not None:
            img = self.distortion(img)
        return self.transform(img), r["label"]

    def counts(self):
        out = {}
        for r in self.rows:
            key = (r.get("generator", "?"), "fake" if r["label"] else "real")
            out[key] = out.get(key, 0) + 1
        return out


def make_loader(dataset, batch_size=64, shuffle=False, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "genimage" / "index.csv"
    for model in ("resnet", "clip"):
        ds = RealFakeDataset(src, model=model, train=True)
        x, y = next(iter(make_loader(ds, batch_size=8, shuffle=True, num_workers=0)))
        print(f"{model}: {len(ds)} images, batch {tuple(x.shape)}, "
              f"mean {x.mean():.3f} std {x.std():.3f}, labels {y.tolist()}")
    for (g, c), n in sorted(ds.counts().items()):
        print(f"  {g:12s} {c:5s} {n}")

    img = Image.open(random.choice(ds.rows)["path"]).convert("RGB")
    for d in (jpeg(50), downscale(0.5), center_crop(0.8)):
        print(f"  {d.__name__}: {img.size} -> {d(img).size}")
