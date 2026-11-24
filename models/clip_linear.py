"""Frozen CLIP + linear classifier (issue #10).

The CLIP image encoder is never trained. Every image goes through it once to
get an embedding, and the only thing we learn is one linear layer on top of
those embeddings. This is the "linear probe" from Ojha et al. (CVPR 2023), who
found CLIP features generalize to unseen generators much better than a
fine-tuned CNN.

The head outputs 2 logits (0 = real, 1 = fake), same as the ResNet-50 model,
so the eval code can treat both models the same way. Inputs come from
RealFakeDataset(model="clip"), which already does the 224x224 resize/crop and
CLIP normalization.

    model = load_clip_linear("checkpoints/clip_linear_best.pth", device)
    probs = model(x).softmax(1)[:, 1]   # P(fake)
"""

import os
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch import nn

LABELS = {0: "real", 1: "fake"}
DEFAULT_ARCH = "ViT-L-14"
DEFAULT_PRETRAINED = "openai"


class CLIPLinear(nn.Module):
    def __init__(self, arch=DEFAULT_ARCH, pretrained=DEFAULT_PRETRAINED, num_classes=2):
        super().__init__()
        clip = open_clip.create_model(arch, pretrained=pretrained)
        self.visual = clip.visual            # drop the text tower, we don't need it
        self.visual.requires_grad_(False)
        self.visual.eval()
        self.arch, self.pretrained = arch, pretrained
        self.embed_dim = self.visual.output_dim
        self.head = nn.Linear(self.embed_dim, num_classes)

    def train(self, mode=True):
        # the encoder stays in eval mode no matter what
        super().train(mode)
        self.visual.eval()
        return self

    @torch.no_grad()
    def embed(self, x):
        """L2-normalized CLIP image embedding, shape (batch, embed_dim)."""
        return F.normalize(self.visual(x).float(), dim=-1)

    def forward(self, x):
        return self.head(self.embed(x))


def head_from_logistic(weight, bias):
    """Turn a binary logistic regression (one weight vector + bias, positive =
    fake) into a 2-logit linear layer. Logit 0 is fixed at 0 and logit 1 is the
    LR score, so softmax(...)[1] is exactly the LR's sigmoid probability."""
    weight = torch.as_tensor(weight, dtype=torch.float32).reshape(-1)
    head = nn.Linear(weight.numel(), 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[1] = weight
        head.bias[1] = float(bias)
    return head


def save_checkpoint(path, model, **extra):
    """Only the head is saved; the CLIP weights get re-downloaded by open_clip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "head_state": model.head.state_dict(),
        "arch": model.arch,
        "pretrained": model.pretrained,
        "embed_dim": model.embed_dim,
        "label_map": LABELS,
        "architecture": "clip_linear",
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_clip_linear(path, device="cpu"):
    """Rebuild the full model (frozen CLIP + trained head) from a checkpoint.
    Returns the model in eval mode on `device`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CLIPLinear(ckpt["arch"], ckpt["pretrained"])
    model.head.load_state_dict(ckpt["head_state"])
    return model.to(device).eval()
