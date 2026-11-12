# Ai-Generated-Content-Detector
A lightweight detector for AI-generated images and video on social media — compares a fine-tuned ResNet-50 against a CLIP-feature classifier, tested for cross-generator generalization and robustness to compression.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

### GenImage subset

```bash
python scripts/download_genimage.py                  # SD v1.4, Midjourney, BigGAN, 3000 real + 3000 fake each
python scripts/download_genimage.py --per-class 1000 # smaller if you're short on disk
```

This pulls from the [nebula/GenImage-arrow](https://huggingface.co/datasets/nebula/GenImage-arrow) mirror on Hugging Face (GenImage test set, split by generator) and only downloads the shards it needs. Images end up in `data/genimage/<generator>/{real,fake}/` with an index at `data/genimage/index.csv` (`path,label,generator,source_name`). label is 0 = real, 1 = fake.

Couple things to keep in mind:
- The "real" half of every GenImage generator is ImageNet val photos, so the same real photo can show up under more than one generator. Dedupe on `source_name` when splitting so it doesn't leak between train and test.
- Real images are JPEG and the fakes are PNG, so a model can cheat by picking up on JPEG artifacts. The robustness tests (JPEG compression on everything) should catch that.

### CIFAKE

```bash
python scripts/check_cifake.py            # audit CIFAKE (temp ~110 MB download, deleted after)
python scripts/check_cifake.py --seed 7   # different visual sample
```

Downloads the public [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) zip to a temp file (no Kaggle login needed), reads the images straight from the zip without extracting them, then deletes it. It counts REAL/FAKE per split, opens every image, looks for duplicates, writes 20 REAL + 20 FAKE contact sheets to `docs/cifake_samples/`, and saves the results to `docs/cifake_audit.md`. Only needs Pillow (`pip install pillow`). Results: 50k/50k train, 10k/10k test, no corrupt files.

## Preprocessing

`models/dataset.py` has one `RealFakeDataset` that both models use. It resizes to 224x224 and normalizes with ImageNet stats for ResNet or CLIP stats for CLIP.

```python
from models.dataset import RealFakeDataset, make_loader, jpeg

train = RealFakeDataset("data/genimage/index.csv", model="resnet", train=True, exclude=["midjourney"])
heldout = RealFakeDataset("data/genimage/index.csv", model="clip", generators=["midjourney"])
compressed = RealFakeDataset("data/genimage/index.csv", model="clip", distortion=jpeg(50))
loader = make_loader(train, batch_size=64, shuffle=True)
```

It takes an index CSV, a folder (anything with `real/` and `fake/` subfolders, including CIFAKE's `REAL/FAKE`), or a list of row dicts. If the CSV has a `split` column you can filter with `split="train"`. There's also `downscale(scale)` and `center_crop(frac)` for the robustness tests.

Sanity check: `python -m models.dataset data/genimage/index.csv`
