"""Grouped rice-field splits, semantic labels, and a pretrained-encoder segmenter."""
from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "rice"
OUT = ROOT / "outputs"
WIDTH, HEIGHT = 384, 288
SEED = 23
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
CLASSES = ["soil", "crop", "weed"]


def image_ids():
    # The paper cut each of 28 original photographs into eight adjacent tiles.
    # Keep all eight tiles together to avoid spatial leakage across splits.
    groups = list(range(28))
    random.Random(SEED).shuffle(groups)
    chosen = {"train": groups[:18], "val": groups[18:23], "test": groups[23:]}
    return {name: [f"{group * 8 + offset + 1:03d}" for group in ids for offset in range(8)] for name, ids in chosen.items()}


@lru_cache(maxsize=1)
def source_fingerprint():
    paths = sorted((DATA / "masks").glob("*.png"))
    digest = hashlib.sha256()
    digest.update(b"rice-figshare-7488830-label-map-v1-grouped-eight-tiles")
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def load_scene(source_id: str):
    image = cv2.cvtColor(cv2.imread(str(DATA / "images" / f"{source_id}.jpg")), cv2.COLOR_BGR2RGB)
    labels = cv2.imread(str(DATA / "masks" / f"{source_id}.png"), cv2.IMREAD_GRAYSCALE)
    instances = []
    for cls, name in ((1, "crop"), (2, "weed")):
        _, component_ids, components, _ = cv2.connectedComponentsWithStats((labels == cls).astype(np.uint8), 8)
        for index, (_, _, _, _, area) in enumerate(components[1:], start=1):
            if area >= 30:
                instances.append({"id": f"{source_id}-{name[0]}{index}", "type": name, "mask": component_ids == index})
    return image, labels, instances


def to_tensor(image):
    arr = (image.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class FieldImages(Dataset):
    def __init__(self, ids, augment=False):
        self.scenes = [(source_id, *load_scene(source_id)[:2]) for source_id in ids]
        self.augment = augment

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        _, image, labels = self.scenes[index]
        if self.augment:
            if random.random() < .5:
                image, labels = np.flip(image, 1), np.flip(labels, 1)
            if random.random() < .5:
                image, labels = np.flip(image, 0), np.flip(labels, 0)
            if random.random() < .5:
                gain = random.uniform(.85, 1.15)
                bias = random.uniform(-12, 12)
                image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        return to_tensor(image), torch.from_numpy(labels.copy()).long()


class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False), nn.GroupNorm(8, outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False), nn.GroupNorm(8, outgoing), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CropWeedNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.up3 = Block(512 + 256, 256)
        self.up2 = Block(256 + 128, 128)
        self.up1 = Block(128 + 64, 64)
        self.up0 = Block(64 + 64, 64)
        self.head = nn.Conv2d(64, 3, 1)

    def forward(self, image):
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.up3(torch.cat((F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False), x3), 1))
        x = self.up2(torch.cat((F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2), 1))
        x = self.up1(torch.cat((F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1), 1))
        x = self.up0(torch.cat((F.interpolate(x, size=x0.shape[-2:], mode="bilinear", align_corners=False), x0), 1))
        return F.interpolate(self.head(x), size=image.shape[-2:], mode="bilinear", align_corners=False)


def loss_fn(logits, labels):
    ce = F.cross_entropy(logits, labels, weight=torch.tensor([.4, 2., 2.], device=logits.device), ignore_index=255)
    probability = F.softmax(logits, dim=1)
    valid = labels != 255
    dice = 0.
    for cls in (1, 2):
        truth = ((labels == cls) & valid).float()
        pred = probability[:, cls] * valid
        dice += 1 - (2 * (pred * truth).sum() + 1) / (pred.sum() + truth.sum() + 1)
    return ce + .3 * dice / 2


def predict(model, image, device):
    model.eval()
    with torch.inference_mode():
        logits = model(to_tensor(image).unsqueeze(0).to(device))
        return logits.softmax(1)[0].cpu().numpy().transpose(1, 2, 0)
