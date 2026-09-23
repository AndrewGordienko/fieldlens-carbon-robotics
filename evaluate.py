"""Held-out segmentation benchmark and crop-overlap stress test."""
from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from segmentation import CropWeedNet, DATA, HEIGHT, OUT, SEED, WIDTH, image_ids, load_scene, predict, source_fingerprint

LEVELS = [0, .2, .4, .6, .8]


def crop_sprite(image, mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    return image[ys.min():ys.max()+1, xs.min():xs.max()+1], mask[ys.min():ys.max()+1, xs.min():xs.max()+1]


def composite(image, label, weed_mask, sprite, target):
    """Choose crop scale to approximate target covered weed fraction, then relabel visible pixels."""
    if target == 0:
        return image.copy(), label.copy(), weed_mask.copy(), np.zeros_like(weed_mask), 0.
    sprite_image, sprite_mask = sprite
    ys, xs = np.nonzero(weed_mask)
    if len(xs) < 10:
        return image.copy(), label.copy(), weed_mask.copy(), np.zeros_like(weed_mask), 0.
    cx, cy = int(xs.mean()), int(ys.mean())
    best = None
    for scale in np.linspace(.16, 3.4, 50):
        height, width = sprite_mask.shape
        sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
        sprite_resized = cv2.resize(sprite_image, (sw, sh), interpolation=cv2.INTER_LINEAR)
        alpha_resized = cv2.resize(sprite_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
        x0, y0 = cx - sw // 2, cy - sh // 2
        left, top, right, bottom = max(0, x0), max(0, y0), min(WIDTH, x0 + sw), min(HEIGHT, y0 + sh)
        if left >= right or top >= bottom:
            continue
        alpha = np.zeros_like(weed_mask)
        alpha[top:bottom, left:right] = alpha_resized[top-y0:bottom-y0, left-x0:right-x0]
        actual = float(np.count_nonzero(alpha & weed_mask) / max(1, weed_mask.sum()))
        error = abs(actual - target)
        if best is None or error < best[0]:
            best = (error, actual, alpha, sprite_resized, (left, top, right, bottom, x0, y0))
    _, actual, alpha, pixels, (left, top, right, bottom, x0, y0) = best
    output = image.copy()
    local = alpha[top:bottom, left:right]
    tile = output[top:bottom, left:right]
    tile[local] = pixels[top-y0:bottom-y0, left-x0:right-x0][local]
    output_label = label.copy()
    output_label[alpha] = 1
    return output, output_label, weed_mask & ~alpha, alpha, actual


def confusion(prob, label, threshold):
    # Treat weed probability as the action score; choose crop/soil only below threshold.
    prediction = np.where(prob[:, :, 2] >= threshold, 2, prob[:, :, :2].argmax(2))
    valid = label != 255
    return np.bincount((label[valid] * 3 + prediction[valid]).ravel(), minlength=9).reshape(3, 3)


def iou(matrix, cls):
    return float(matrix[cls, cls] / max(1, matrix[cls].sum() + matrix[:, cls].sum() - matrix[cls, cls]))


def save_visuals(case_id, image, label, probability, threshold, target_mask):
    directory = OUT / "images"
    directory.mkdir(exist_ok=True)
    palette = np.array([[97, 82, 74], [67, 145, 222], [231, 100, 115]], np.uint8)
    truth = palette[np.minimum(label, 2)]
    truth[label == 255] = [178, 166, 212]
    predicted_classes = np.where(probability[:, :, 2] >= threshold, 2, probability[:, :, :2].argmax(2))
    predicted = palette[predicted_classes]
    heat = (np.clip(probability[:, :, 2], 0, 1) * 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
    ys, xs = np.nonzero(target_mask)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    # Show the selected component at a useful scale while retaining nearby crop context.
    crop_width = min(WIDTH, max(160, int(bx1 - bx0 + 45), int((by1 - by0 + 45) * 4 / 3)))
    crop_height = min(HEIGHT, round(crop_width * 3 / 4))
    crop_x = int(np.clip((bx0 + bx1) // 2 - crop_width // 2, 0, WIDTH - crop_width))
    crop_y = int(np.clip((by0 + by1) // 2 - crop_height // 2, 0, HEIGHT - crop_height))
    views = {
        "rgb": image,
        "gt": (.58 * image + .42 * truth).astype(np.uint8),
        "pred": (.58 * image + .42 * predicted).astype(np.uint8),
        "heat": (.45 * image + .55 * heat).astype(np.uint8),
    }
    for name, pixels in views.items():
        detail = Image.fromarray(pixels[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]).resize((768, 576), Image.Resampling.BICUBIC)
        draw = ImageDraw.Draw(detail)
        sx, sy = 768 / crop_width, 576 / crop_height
        box = ((bx0-crop_x)*sx, (by0-crop_y)*sy, (bx1-crop_x)*sx, (by1-crop_y)*sy)
        draw.rectangle(box, outline="#ffffff", width=6)
        draw.rectangle(box, outline="#5069ee", width=3)
        detail.save(directory / f"{case_id}-{name}.jpg", quality=91)
    return {name: f"images/{case_id}-{name}.jpg" for name in ("rgb", "gt", "pred", "heat")}


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    image_output = OUT / "images"
    image_output.mkdir(exist_ok=True)
    for previous in image_output.glob("*.jpg"):
        previous.unlink()
    checkpoint = torch.load(OUT / "model.pt", map_location="cpu", weights_only=False)
    if checkpoint["fingerprint"] != source_fingerprint():
        raise SystemExit("Dataset annotation fingerprint changed; retrain before evaluation")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = CropWeedNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    splits = image_ids()
    validation = [(sid, *load_scene(sid)) for sid in splits["val"]]
    testing = [(sid, *load_scene(sid)) for sid in splits["test"]]
    val_predictions = [(sid, label, predict(model, image, device)) for sid, image, label, _ in validation]
    # Conservative pixel threshold: best validation weed recall while crop-to-weed error <= 1%.
    choices = []
    for threshold in np.arange(.3, .951, .025):
        crop_pixels = sum(int((label == 1).sum()) for _, label, _ in val_predictions)
        weed_pixels = sum(int((label == 2).sum()) for _, label, _ in val_predictions)
        crop_errors = sum(int(((label == 1) & (prob[:, :, 2] >= threshold)).sum()) for _, label, prob in val_predictions)
        weed_hits = sum(int(((label == 2) & (prob[:, :, 2] >= threshold)).sum()) for _, label, prob in val_predictions)
        choices.append((float(threshold), crop_errors / max(1, crop_pixels), weed_hits / max(1, weed_pixels)))
    feasible = [row for row in choices if row[1] <= .01]
    threshold = min(feasible or choices, key=lambda row: -row[2])[0]
    start = time.perf_counter()
    test_predictions = [(sid, image, label, instances, predict(model, image, device)) for sid, image, label, instances in testing]
    latency_ms = (time.perf_counter() - start) * 1000 / len(testing)
    matrix = sum((confusion(prob, label, threshold) for _, _, label, _, prob in test_predictions), np.zeros((3, 3), np.int64))
    crop_pixels = sum(int((label == 1).sum()) for _, _, label, _, _ in test_predictions)
    weed_pixels = sum(int((label == 2).sum()) for _, _, label, _, _ in test_predictions)
    clean_crop_error = sum(int(((label == 1) & (prob[:, :, 2] >= threshold)).sum()) for _, _, label, _, prob in test_predictions) / max(1, crop_pixels)
    clean_weed_recall = sum(int(((label == 2) & (prob[:, :, 2] >= threshold)).sum()) for _, _, label, _, prob in test_predictions) / max(1, weed_pixels)
    per_image = []
    for sid, _, label, _, prob in test_predictions:
        cm = confusion(prob, label, threshold)
        n_crop, n_weed = int((label == 1).sum()), int((label == 2).sum())
        per_image.append({
            "source": sid, "crop_pixels": n_crop, "weed_pixels": n_weed,
            "crop_false_pixels": int(((label == 1) & (prob[:, :, 2] >= threshold)).sum()),
            "weed_hit_pixels": int(((label == 2) & (prob[:, :, 2] >= threshold)).sum()),
            "crop_iou": iou(cm, 1), "weed_iou": iou(cm, 2),
        })
    for row in per_image:
        row["crop_false_weed_rate"] = row["crop_false_pixels"] / max(1, row["crop_pixels"])
        row["weed_pixel_recall"] = row["weed_hit_pixels"] / max(1, row["weed_pixels"])
    operating_curve = []
    for choice in choices:
        t = choice[0]
        crop_error = sum(int(((label == 1) & (prob[:, :, 2] >= t)).sum()) for _, _, label, _, prob in test_predictions) / max(1, crop_pixels)
        weed_recall = sum(int(((label == 2) & (prob[:, :, 2] >= t)).sum()) for _, _, label, _, prob in test_predictions) / max(1, weed_pixels)
        operating_curve.append({"threshold": t, "val_crop_error": choice[1], "val_weed_recall": choice[2], "test_crop_error": crop_error, "test_weed_recall": weed_recall})
    bootstrap = random.Random(SEED)
    boot_crop, boot_weed = [], []
    for _ in range(2000):
        sample = [bootstrap.choice(per_image) for _ in per_image]
        boot_crop.append(sum(r["crop_false_pixels"] for r in sample) / max(1, sum(r["crop_pixels"] for r in sample)))
        boot_weed.append(sum(r["weed_hit_pixels"] for r in sample) / max(1, sum(r["weed_pixels"] for r in sample)))
    # A plant-pixel binary calibration check, kept separate from model training/threshold choice.
    plant_prob = np.concatenate([prob[:, :, 2][(label == 1) | (label == 2)] for _, _, label, _, prob in test_predictions])
    plant_truth = np.concatenate([(label[(label == 1) | (label == 2)] == 2).astype(np.float32) for _, _, label, _, _ in test_predictions])
    brier = float(np.mean((plant_prob - plant_truth) ** 2))
    calibration = []
    for low in np.arange(0, 1, .1):
        selected = (plant_prob >= low) & (plant_prob < low + .1)
        calibration.append({"bin": float(low), "count": int(selected.sum()), "confidence": float(plant_prob[selected].mean()) if selected.any() else None, "accuracy": float(plant_truth[selected].mean()) if selected.any() else None})
    train_crops = []
    for sid in splits["train"]:
        image, _, instances = load_scene(sid)
        train_crops.extend(sprite for item in instances if item["type"] == "crop" if (sprite := crop_sprite(image, item["mask"])) is not None)
    rng = random.Random(SEED)
    by_image = {sid: [item["id"] for item in instances if item["type"] == "weed" and item["mask"].sum() >= 30] for sid, _, _, instances, _ in test_predictions}
    selected_weeds = {rng.choice(eligible) for eligible in by_image.values() if eligible}
    available_weeds = [item for eligible in by_image.values() for item in eligible if item not in selected_weeds]
    rng.shuffle(available_weeds)
    selected_weeds.update(available_weeds[:max(0, 60 - len(selected_weeds))])
    rows = []
    gallery = []
    for sid, image, label, instances, clean_prob in test_predictions:
        for instance in instances:
            if instance["id"] not in selected_weeds:
                continue
            sprite = rng.choice(train_crops)
            for level in LEVELS:
                altered, altered_label, visible, overlay, actual = composite(image, label, instance["mask"], sprite, level)
                prob = clean_prob if level == 0 else predict(model, altered, device)
                visible_count = int(visible.sum())
                weed_hit = int(((prob[:, :, 2] >= threshold) & visible).sum())
                crop_false = int(((prob[:, :, 2] >= threshold) & overlay).sum())
                record = {
                    "id": instance["id"], "source": sid, "target": level, "actual": actual,
                    "visible_pixels": visible_count, "weed_hit_pixels": weed_hit,
                    "visible_weed_recall": weed_hit / max(1, visible_count),
                    "mean_weed_probability": float(prob[:, :, 2][visible].mean()) if visible_count else None,
                    "occluder_crop_pixels": int(overlay.sum()), "occluder_false_weed_pixels": crop_false,
                    "instance_detected": bool(visible_count >= 10 and weed_hit / max(1, visible_count) >= .1),
                }
                rows.append(record)
                if level in (0, .4, .8):
                    case_id = f"{instance['id']}-{round(level * 100)}"
                    images = save_visuals(case_id, altered, altered_label, prob, threshold, instance["mask"])
                    gallery.append({**record, "images": images})
    sweep = []
    for level in LEVELS:
        group = [row for row in rows if row["target"] == level]
        visible_total = sum(row["visible_pixels"] for row in group)
        crop_total = sum(row["occluder_crop_pixels"] for row in group)
        sweep.append({
            "target": level, "actual_mean": float(statistics.mean(row["actual"] for row in group)), "cases": len(group),
            "weed_pixel_recall": sum(row["weed_hit_pixels"] for row in group) / max(1, visible_total),
            "instance_recall": sum(row["instance_detected"] for row in group) / len(group),
            "crop_false_weed_rate": sum(row["occluder_false_weed_pixels"] for row in group) / max(1, crop_total),
            "mean_weed_probability": float(statistics.mean(row["mean_weed_probability"] for row in group if row["mean_weed_probability"] is not None)),
        })
    gallery.sort(key=lambda row: (row["target"] != .8, row["visible_weed_recall"], -row["occluder_false_weed_pixels"]))
    result = {
        "project": "Crop/weed semantic segmentation under overlap", "source_fingerprint": source_fingerprint(),
        "model": "ImageNet-pretrained ResNet-18 encoder + U-Net-style decoder", "checkpoint_epoch": checkpoint["epoch"],
        "split": {name: {"images": len(ids)} for name, ids in splits.items()}, "threshold": threshold, "validation_threshold_choices": choices,
        "clean": {"iou_soil": iou(matrix, 0), "iou_crop": iou(matrix, 1), "iou_weed": iou(matrix, 2), "mean_plant_iou": (iou(matrix, 1) + iou(matrix, 2)) / 2, "weed_pixel_recall": clean_weed_recall, "crop_false_weed_rate": clean_crop_error, "brier_plant": brier, "latency_ms_per_image": latency_ms, "confusion": matrix.tolist(), "crop_pixels": crop_pixels, "weed_pixels": weed_pixels},
        "calibration": calibration, "sweep": sweep, "cases": rows, "gallery": gallery, "stress_sample_count": len(selected_weeds),
        "per_image": sorted(per_image, key=lambda r: r["crop_false_weed_rate"], reverse=True),
        "operating_curve": operating_curve,
        "bootstrap_95": {"crop_false_weed_rate": [float(np.quantile(boot_crop, .025)), float(np.quantile(boot_crop, .975))], "weed_pixel_recall": [float(np.quantile(boot_weed, .025)), float(np.quantile(boot_weed, .975))]},
        "illustrative_gate": {"max_crop_false_weed_rate": .01, "min_weed_pixel_recall": .70, "passed": bool(clean_crop_error <= .01 and clean_weed_recall >= .70)},
        "limitations": ["Crop overlap is composited from separately photographed training crops.", "The rice-field dataset has 28 original photos; no Carbon Robotics images or models were used.", "Crop-to-weed pixel rate is a vision risk proxy, not measured laser/crop damage.", "Instance detection uses a 10% visible-pixel rule and is not a meristem targeting metric."],
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"clean": result["clean"], "sweep": sweep, "gallery_count": len(gallery)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
