"""Train the crop/weed semantic segmenter and save the best validation checkpoint."""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from segmentation import CropWeedNet, FieldImages, OUT, SEED, image_ids, loss_fn, source_fingerprint


def confusion_matrix(model, loader, device):
    matrix = np.zeros((3, 3), np.int64)
    model.eval()
    with torch.inference_mode():
        for image, label in loader:
            predicted = model(image.to(device)).argmax(1).cpu().numpy()
            truth = label.numpy()
            good = truth != 255
            matrix += np.bincount((truth[good] * 3 + predicted[good]).ravel(), minlength=9).reshape(3, 3)
    return matrix


def iou(matrix):
    return [float(matrix[k, k] / max(1, matrix[k].sum() + matrix[:, k].sum() - matrix[k, k])) for k in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OUT.mkdir(exist_ok=True)
    ids = image_ids()
    train = FieldImages(ids["train"], augment=True)
    val = FieldImages(ids["val"])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=2, num_workers=0)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = CropWeedNet(pretrained=True).to(device)
    encoder = list(model.stem.parameters()) + list(model.layer1.parameters()) + list(model.layer2.parameters()) + list(model.layer3.parameters()) + list(model.layer4.parameters())
    decoder = list(model.up3.parameters()) + list(model.up2.parameters()) + list(model.up1.parameters()) + list(model.up0.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": 5e-5}, {"params": decoder, "lr": 3e-4}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best = -1.
    start = time.time()
    print(f"Training on {device}; {len(train)} train and {len(val)} validation field images", flush=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        matrix = confusion_matrix(model, val_loader, device)
        scores = iou(matrix)
        score = (scores[1] + scores[2]) / 2
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_iou_soil": scores[0], "val_iou_crop": scores[1], "val_iou_weed": scores[2], "val_mean_plant_iou": score}
        history.append(record)
        if score > best:
            best = score
            torch.save({"model": model.cpu().state_dict(), "epoch": epoch + 1, "seed": SEED, "fingerprint": source_fingerprint(), "validation_score": score}, OUT / "model.pt")
            model.to(device)
        print(json.dumps(record), flush=True)
    metadata = {"device": device, "duration_seconds": round(time.time() - start, 1), "epochs": args.epochs, "best_val_mean_plant_iou": best, "split": ids, "source_fingerprint": source_fingerprint(), "history": history}
    (OUT / "training.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {OUT / 'model.pt'}; best validation plant mIoU={best:.3f}", flush=True)


if __name__ == "__main__":
    main()
