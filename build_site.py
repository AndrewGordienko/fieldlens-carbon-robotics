"""Package the evaluated run as a zero-backend static site for GitHub Pages."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from segmentation import OUT, ROOT

SITE = ROOT / "docs"


def main():
    report = json.loads((OUT / "results.json").read_text())
    exported = json.loads((OUT / "export.json").read_text())
    if report["source_fingerprint"] != exported["source_fingerprint"]:
        raise SystemExit("Report and exported model have different dataset fingerprints")
    if not report["gallery"]:
        raise SystemExit("No evaluated image cases available")
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()
    shutil.copy(ROOT / "static" / "index.html", SITE / "index.html")
    shutil.copytree(ROOT / "static", SITE / "static", ignore=shutil.ignore_patterns("index.html"))
    (SITE / "results.json").write_text(json.dumps(report, separators=(",", ":")))
    images = SITE / "images"
    images.mkdir()
    referenced = {Path(path).name for case in report["gallery"] for path in case["images"].values()}
    for name in referenced:
        shutil.copy(OUT / "images" / name, images / name)
    (SITE / ".nojekyll").write_text("")
    (SITE / "LICENSE-DATA.txt").write_text("Rice seedling and weed field images: Ma et al. (2019), CC BY 4.0. Source: https://figshare.com/articles/dataset/rice_seedlings_and_weeds/7488830 . Images have been resized and combined with model overlays for this project.\n")
    print(f"Built {SITE}: {len(referenced)} inspected image views and one evaluated report")


if __name__ == "__main__":
    main()
