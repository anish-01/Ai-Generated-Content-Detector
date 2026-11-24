# In-distribution evaluation report (issue #13)

**Task:** run both models (fine-tuned ResNet-50 and the frozen-CLIP linear classifier)
on the in-distribution test set and record accuracy and ROC-AUC for each.

**Status: partially complete.** ResNet-50 was evaluated on the full test set.
The CLIP linear classifier could not be evaluated because its trained checkpoint
(`checkpoints/clip_linear_best.pth`) is not available locally (see
[Blockers](#blockers-and-limitations)). No CLIP numbers are reported.

## Results

| model | status | test images | accuracy | ROC-AUC | checkpoint |
|---|---|---|---|---|---|
| ResNet-50 (fine-tuned) | evaluated | 18,148 | **98.15%** (0.981486) | **0.9984** (0.998427) | `checkpoints/resnet50_best.pth` (epoch 3) |
| CLIP ViT-L/14 + linear head | **blocked** | - | - | - | `checkpoints/clip_linear_best.pth` (missing) |

Source: `results/in_distribution_evaluation.csv`. A comparison between the two
models is not possible until CLIP is run on the same 18,148 images.

### ResNet-50 details

Confusion matrix (threshold P(fake) >= 0.5, i.e. argmax of the two logits):

| | predicted REAL | predicted FAKE |
|---|---|---|
| **REAL (0)** | 8,954 | 120 |
| **FAKE (1)** | 216 | 8,858 |

- REAL recall 98.68%, FAKE recall 97.62%, F1 (fake) 0.9814.
- Slightly more fakes are missed (216) than reals are flagged (120).

Accuracy by source:

| source / generator | images | accuracy |
|---|---|---|
| CIFAKE | 18,000 | 98.18% |
| GenImage BigGAN | 74 | 94.59% |
| GenImage SD v1.4 | 74 | 93.24% |

The GenImage subsets are tiny (74 images each, so one image is about 1.35
points), and the overall number is almost entirely CIFAKE. For reference, the
checkpoint's validation accuracy at epoch 3 was 98.28%, close to the 98.15%
measured here on the test set. The validation number was not used as a test
result.

## Dataset

`splits/test.csv` from issue #2 (seed 42, 70/15/15 split), checked again for this task:

| check | result |
|---|---|
| rows | 18,148 (matches the issue #2 report) |
| REAL / FAKE | 9,074 / 9,074 |
| CIFAKE / GenImage BigGAN / GenImage SD v1.4 | 18,000 / 74 / 74 |
| labels only 0 or 1, `split` column all `test` | yes |
| files present | 18,148 / 18,148 (0 missing) |
| files that decode with PIL | 18,148 (0 corrupt) |
| MD5 matches the manifest | 18,148 (0 mismatched) |
| overlap with `cross_generator_test.csv` / any Midjourney image | 0 / 0 |
| `scripts/split_datasets.py --verify-only` | all checks passed |

The cross-generator holdout (`splits/cross_generator_test.csv`, Midjourney) is
excluded: the script refuses that file and any row whose `split` isn't `test`.
The test set was not changed, and no images were substituted.

## Models and checkpoints

**ResNet-50** (`models/resnet50.py`, issue #9): torchvision ResNet-50 with the fc
layer replaced by `Linear(2048, 2)`. Fine-tuned from `IMAGENET1K_V2` on
`splits/train.csv` (84,700 images) for 3 epochs, batch size 32, lr 1e-4,
weight decay 1e-4, seed 42, trained on Kaggle (CUDA).

Checkpoint `checkpoints/resnet50_best.pth` (MD5 `288210d5e658ede62e83fd9bb3c8ce64`):

- loads with `torch.load(..., weights_only=True)` (no arbitrary pickled code)
- `architecture: resnet50`, `num_classes: 2`, `label_map: {0: 'real', 1: 'fake'}` (matches the dataset)
- `fc.weight` is (2, 2048); all 320 tensors load into the architecture with `strict=True`
- epoch 3 (best by val loss 0.0474)

`checkpoints/resnet50_last.pth` is the final-epoch ResNet-50 checkpoint (same
epoch and metrics as best, since epoch 3 was the best). It is not a CLIP model
and was not used.

**CLIP linear classifier** (`models/clip_linear.py`, issue #10): frozen OpenAI
CLIP ViT-L/14 image encoder, L2-normalized 768-d embedding, and a 2-logit linear
head built from a logistic regression (best C = 100 by val loss). The evaluation
script loads it with the project's own `CLIPLinear` class and checks
`architecture == 'clip_linear'`, the label map and the head shape (2 x embed_dim).

## Methodology

- `scripts/evaluate_in_distribution.py` loads `splits/test.csv` once and gives
  the same ordered row list to every model. A fingerprint (MD5 of the ordered
  path/label list) is stored per model, and the script fails if two models'
  fingerprints differ.
- Preprocessing reuses `models/dataset.RealFakeDataset` with `train=False`
  (bicubic resize to 224, center crop 224), using ImageNet normalization for
  ResNet-50 and CLIP normalization for CLIP, the same as in training.
- `model.eval()` + `torch.inference_mode()`. Batch size 64, 4 loader workers,
  no shuffling.
- `P(fake) = torch.softmax(logits, dim=1)[:, 1]`; predicted label = argmax of the logits.
- Metrics: `sklearn.metrics.accuracy_score` on predicted labels;
  `roc_auc_score` and `roc_curve` on the continuous P(fake). Before computing
  them the script checks that both classes are present, scores are finite and
  in [0, 1], predicted labels are in {0, 1}, the prediction count equals the
  image count, and the loader labels match the manifest order.
- Device is selected automatically: CUDA, then Apple MPS, then CPU (`--device`
  overrides it). This run used MPS (Apple Silicon) with torch 2.15.0 and
  scikit-learn 1.9.1, and took 409 s for 18,148 images.

## Output files

| file | content |
|---|---|
| `results/in_distribution_evaluation.csv` | one row per model: status, counts, accuracy, ROC-AUC, checkpoint path + MD5 + epoch, test fingerprint, device, notes |
| `results/in_distribution_predictions.csv` | 18,148 rows (ResNet-50 only): image_path, model, label, pred_label, p_fake |
| `results/in_distribution_roc_curves.csv` | ROC curve points (fpr, tpr, threshold) for each evaluated model |

The saved numbers were checked independently by recomputing accuracy (0.981486)
and ROC-AUC (0.998427) from the predictions file. The image set in the
predictions file is exactly the 18,148 paths in `splits/test.csv`.

## Automated tests

`python -m unittest tests.test_evaluate_in_distribution tests.test_split_datasets -v`
gives **48 tests, all passed** (34 new + 14 existing for issue #2).

New tests (`tests/test_evaluate_in_distribution.py`):

- metrics: known accuracy and ROC-AUC values; ROC-AUC uses continuous scores
  (perfect ranking with all scores > 0.5 gives AUC 1.0 while accuracy is 0.5);
  rejects a single class, NaN, scores outside [0, 1] and labels outside {0, 1}
- checkpoints: missing file, directory, wrong architecture, swapped label map,
  3-output head, incomplete weights, and a ResNet checkpoint passed as CLIP all
  raise a clear `CheckpointError`; the real checkpoint loads, is in eval mode
  and outputs shape (N, 2)
- manifest: the cross-generator holdout and non-`test` rows are refused;
  invalid labels and missing images are reported; the real manifest has 18,148
  rows (9,074/9,074) and no Midjourney/holdout images
- pipeline: prediction count and order match the manifest, model outputs that
  aren't 2 logits are rejected, both models get the identical image list, the
  saved CSV metrics match metrics recomputed from the predictions, a missing
  model is recorded as `blocked` with empty metrics and no predictions, and
  the saved real results match their predictions file

The synthetic tests use tiny generated images and stand-in models only to check
the code. Their numbers are not model results.

## Blockers and limitations

1. **CLIP checkpoint missing (blocks CLIP evaluation).** The issue #10 training
   script (`scripts/train_clip_linear.py` on `main`) writes
   `checkpoints/clip_linear_best.pth`, which is git-ignored and so not in the
   repo. It was not found in `checkpoints/`, anywhere under the project, or
   anywhere in the home directory (Spotlight search). There is also no cached
   `features/` directory to rebuild it from. Only `results/clip_linear_c_sweep.csv`
   (validation metrics) is on `main`. CLIP was not retrained and no substitute
   model was used.
2. **Branch is behind `main`.** This branch was created before PRs #32 (ResNet-50)
   and #33 (CLIP) were merged, so `models/resnet50.py` and `models/clip_linear.py`
   were missing here. They were copied byte-for-byte from `origin/main` so the
   evaluation can reuse them. Merging `main` into this branch should bring them
   in without conflicts.
3. **Low disk space.** About 200 MB was free during this run. The CLIP run needs
   `open_clip_torch` plus about 0.9 GB of ViT-L/14 weights that open_clip
   downloads.
4. The test set is about 99% CIFAKE (32x32 images upscaled to 224), so these
   numbers mostly measure CIFAKE performance. The GenImage per-generator
   accuracies come from 74 images each and are noisy. GenImage reals are JPEG
   and fakes are PNG, a known shortcut risk (see README).

### Next steps to finish CLIP

1. Get `clip_linear_best.pth` from whoever ran issue #10 and put it at
   `checkpoints/clip_linear_best.pth`, **or**, with team approval, retrain it:
   `python scripts/train_clip_linear.py` (from `main`; about 2.5 h to extract
   ViT-L/14 features on an M2, never touches the test split).
2. Free about 1.5 GB of disk and run `pip install open_clip_torch`.
3. Rerun the evaluation (command below). Both models will then be evaluated on
   the same 18,148 images and the comparison table above can be filled in.

## Reproduce

```bash
source .venv/bin/activate
pip install -r requirements.txt            # needs scikit-learn; open_clip_torch for CLIP
python scripts/split_datasets.py --verify-only
python scripts/evaluate_in_distribution.py                      # both models; missing ones -> "blocked"
python scripts/evaluate_in_distribution.py --models resnet50 --batch-size 64
python scripts/evaluate_in_distribution.py --clip-checkpoint path/to/clip_linear_best.pth
python -m unittest tests.test_evaluate_in_distribution tests.test_split_datasets -v
```

Options: `--test-csv`, `--resnet-checkpoint`, `--clip-checkpoint`, `--models`,
`--batch-size`, `--workers`, `--device {auto,cuda,mps,cpu}`, `--results-dir`,
`--limit N` (seeded smoke-test subset only; don't write it to `results/`).
