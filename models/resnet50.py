"""ResNet-50 real-vs-generated classifier + checkpoint helpers (issue #9).

ImageNet-pretrained torchvision ResNet-50 with the 1000-way fc layer swapped
for a 2-logit head (0 = real, 1 = fake). Train with nn.CrossEntropyLoss.
Inputs come from RealFakeDataset(model="resnet"), which already applies the
224x224 resize/crop and ImageNet normalization these weights expect.
"""

import os
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

LABELS = {0: "real", 1: "fake"}
WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


def build_resnet50(pretrained=True, num_classes=2):
    model = resnet50(weights=WEIGHTS if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(path, model, optimizer=None, scheduler=None, **extra):
    """Write the state dicts plus anything in `extra` (epoch, metrics, config...).
    Written to a temp file first so an interrupted save can't corrupt the
    previous checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "label_map": LABELS,
        "architecture": "resnet50",
        "num_classes": model.fc.out_features,
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, device="cpu"):
    """Load a checkpoint onto `device` (works across cuda/mps/cpu).
    Builds the model if none is given. Returns (model, checkpoint_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if model is None:
        model = build_resnet50(pretrained=False, num_classes=ckpt.get("num_classes", 2))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    if optimizer is not None and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return model, ckpt
