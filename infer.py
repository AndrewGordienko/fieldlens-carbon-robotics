"""Run the exported segmenter on an arbitrary RGB image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from segmentation import HEIGHT, MEAN, OUT, STD, WIDTH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("prediction.png"))
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    report = json.loads((OUT / "results.json").read_text())
    threshold = report["threshold"] if args.threshold is None else args.threshold
    if not 0 <= threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    image = Image.open(args.image).convert("RGB").resize((WIDTH, HEIGHT))
    arr = np.asarray(image).copy()
    tensor = torch.from_numpy(((arr.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy()).unsqueeze(0)
    model = torch.jit.load(str(OUT / "model.ts"), map_location="cpu")
    with torch.inference_mode():
        probability = model(tensor).softmax(1)[0].numpy().transpose(1, 2, 0)
    prediction = np.where(probability[:, :, 2] >= threshold, 2, probability[:, :, :2].argmax(2))
    palette = np.array([[97, 82, 74], [67, 145, 222], [231, 100, 115]], np.uint8)
    overlay = (.55 * arr + .45 * palette[prediction]).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.output)
    mask_path = args.output.with_name(args.output.stem + "-mask.png")
    Image.fromarray(prediction.astype(np.uint8)).save(mask_path)
    print(json.dumps({"overlay": str(args.output), "mask": str(mask_path), "threshold": threshold, "weed_pixels": int((prediction == 2).sum()), "crop_pixels": int((prediction == 1).sum()), "warning": "Research prototype; never use for field actuation."}, indent=2))


if __name__ == "__main__":
    main()
