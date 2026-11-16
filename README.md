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

### FaceForensics++ frames

FF++ has to be requested and downloaded with the official script first. Then pull ~1 frame per second from each video:

```bash
# try it on one video per folder first and check the output + disk estimate
python scripts/extract_ffpp_frames.py --ffpp-root /path/to/FaceForensics --limit 1 --out data/ffpp_test
# full run (resumable: rerun after an interruption and finished videos are skipped)
python scripts/extract_ffpp_frames.py --ffpp-root /path/to/FaceForensics --workers 8
```

Frames go to `data/ffpp/{real,fake}/<source or method>/<video>/<video>_f<frame>.png` with an index at `data/ffpp/index.csv` (`path,label,generator,video,source_video,frame,compression`). Defaults are c23 and lossless PNG; `--format jpg` is much smaller but adds a second round of compression. Split by `source_video`, not by frame. Frames from one clip are near-duplicates, and fake `000_003` shares its background with real `000`.

### CIFAKE

```bash
python scripts/check_cifake.py            # audit CIFAKE (temp ~110 MB download, deleted after)
python scripts/check_cifake.py --seed 7   # different visual sample
```

Downloads the public [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) zip to a temp file (no Kaggle login needed), reads the images straight from the zip without extracting them, then deletes it. It counts REAL/FAKE per split, opens every image, looks for duplicates, writes 20 REAL + 20 FAKE contact sheets to `docs/cifake_samples/`, and saves the results to `docs/cifake_audit.md`. Only needs Pillow (`pip install pillow`). Results: 50k/50k train, 10k/10k test, no corrupt files.

For the train/val/test split the images need to be on disk:

```bash
curl -L -o /tmp/cifake.zip https://www.kaggle.com/api/v1/datasets/download/birdy654/cifake-real-and-ai-generated-synthetic-images
mkdir -p data/cifake && unzip -q /tmp/cifake.zip -d data/cifake && rm /tmp/cifake.zip   # ~0.6 GB on disk
```

## Train / val / test split

```bash
python scripts/download_genimage.py --per-class 250   # small GenImage subset (~0.4 GB)
python scripts/split_datasets.py                      # writes splits/*.csv + docs/dataset_split_report.md
python scripts/split_datasets.py --verify-only        # re-run the leakage checks
python -m unittest tests.test_split_datasets -v
```

CIFAKE + GenImage are split 70/15/15 (seed 42, stratified by dataset/generator/label) into `splits/train.csv`, `val.csv` and `test.csv`. Midjourney is held out completely in `splits/cross_generator_test.csv` for the cross-generator test. Identical files (MD5) and GenImage reals that share an ImageNet `source_name` always stay in one split, and anything that overlaps the held-out generator is dropped from the normal splits (listed in `splits/excluded.csv`). Near-duplicates aren't detected. Details and counts are in `docs/dataset_split_report.md`.

```python
train = RealFakeDataset("splits/train.csv", model="resnet", train=True)
cross = RealFakeDataset("splits/cross_generator_test.csv", model="clip")
```

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

## Models

### CLIP linear probe

Frozen CLIP ViT-L/14 (OpenAI weights) with one linear layer on top, following Ojha et al. 2023. CLIP itself is never trained. Every image gets embedded once, and then a logistic regression is fit on those embeddings.

```bash
python scripts/train_clip_linear.py                    # ViT-L/14, ~2.5 h on an M2 (mostly embedding images)
python scripts/train_clip_linear.py --arch ViT-B-16    # ~30 min, a bit weaker
python scripts/train_clip_linear.py --limit-train 256 --limit-val 128   # smoke test
```

Embeddings get cached in `features/` and saved as it goes, so if it gets interrupted just rerun the same command and it continues where it stopped. After that it tries a few regularization strengths (C) and keeps the one with the lowest val loss. It never touches the test split or the Midjourney holdout.

Outputs: `checkpoints/clip_linear_best.pth` (just the linear head, CLIP weights get downloaded by open_clip), `results/clip_linear_c_sweep.csv` and `.png`.

Using it (outputs 2 logits like the ResNet, 0 = real, 1 = fake):

```python
from models.clip_linear import load_clip_linear
from models.dataset import RealFakeDataset, make_loader

model = load_clip_linear("checkpoints/clip_linear_best.pth", device)
ds = RealFakeDataset("splits/test.csv", model="clip")   # model="clip" for CLIP normalization
for x, y in make_loader(ds):
    p_fake = model(x.to(device)).softmax(1)[:, 1]
```

Val results (ViT-L/14, picked C=100 by val loss):

| | accuracy | ROC-AUC |
|---|---|---|
| all of val (18,152) | 95.74% | 0.9913 |
| CIFAKE (18,000) | 95.77% | |
| GenImage BigGAN (76) | 96.05% | |
| GenImage SD 1.4 (76) | 88.16% | |

The curve basically flattens out past C=3, and train/val stay close the whole way, so it's not overfitting (makes sense, it's only 769 parameters). Heads up that the GenImage val sets are tiny, so those two numbers are noisy. The real test is the Midjourney holdout in #15.
