"""Fetch the CC BY 4.0 rice crop/weed dataset and prepare tiles for this project."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "rice"
URL = "https://ndownloader.figshare.com/files/14519387"
SIZE = (384, 288)


def main():
    if len(list((DEST / "images").glob("*.jpg"))) == 224:
        print(f"Already prepared: {DEST}")
        return
    if not shutil.which("bsdtar"):
        raise SystemExit("bsdtar is required to extract the source RAR archive")
    with tempfile.TemporaryDirectory() as tempdir:
        temp = Path(tempdir)
        archive = temp / "rice-weed.rar"
        print("Downloading 224 labeled rice-field tiles…", flush=True)
        urllib.request.urlretrieve(URL, archive)
        subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(temp)], check=True)
        (DEST / "images").mkdir(parents=True, exist_ok=True)
        (DEST / "masks").mkdir(parents=True, exist_ok=True)
        for number in range(1, 225):
            raw_image = temp / "image" / f"image_{number}.jpg"
            raw_mask = temp / "PixelLabelData" / f"Label_{number}.png"
            image = Image.open(raw_image).convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)
            mask = np.array(Image.open(raw_mask))
            if not set(np.unique(mask)).issubset({0, 1, 2, 3}):
                raise SystemExit(f"Unexpected label values in {raw_mask}")
            # Source codes: 0 = unlabeled, 1 = rice crop, 2 = background, 3 = weed.
            # Local codes: 0 = background, 1 = crop, 2 = weed, 255 = ignore.
            mapped = np.choose(mask, np.array([255, 1, 0, 2], np.uint8))
            mapped = cv2.resize(mapped, SIZE, interpolation=cv2.INTER_NEAREST)
            image.save(DEST / "images" / f"{number:03d}.jpg", quality=92)
            Image.fromarray(mapped).save(DEST / "masks" / f"{number:03d}.png")
    print(f"Prepared 224 tiles in {DEST}. Source: Ma et al. 2019, Figshare 7488830, CC BY 4.0.")


if __name__ == "__main__":
    main()
