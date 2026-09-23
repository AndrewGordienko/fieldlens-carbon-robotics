"""Export a fixed-size TorchScript inference artifact and verify numerical parity."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from segmentation import CropWeedNet, HEIGHT, OUT, WIDTH, source_fingerprint


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(OUT / "model.pt", map_location="cpu", weights_only=False)
    if checkpoint["fingerprint"] != source_fingerprint():
        raise SystemExit("Checkpoint provenance does not match the local dataset")
    model = CropWeedNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    sample = torch.randn(1, 3, HEIGHT, WIDTH)
    traced = torch.jit.trace(model, sample)
    traced = torch.jit.freeze(traced)
    with torch.inference_mode():
        reference = model(sample)
        exported = traced(sample)
    max_abs_error = float((reference - exported).abs().max())
    if max_abs_error > 1e-4:
        raise SystemExit(f"Export parity failed: max absolute logit error {max_abs_error}")
    artifact = OUT / "model.ts"
    traced.save(str(artifact))
    # Include a cold-independent CPU timing for the exported artifact.
    for _ in range(5):
        traced(sample)
    times = []
    with torch.inference_mode():
        for _ in range(30):
            start = time.perf_counter()
            traced(sample)
            times.append((time.perf_counter() - start) * 1000)
    metadata = {"format": "TorchScript", "input_shape": [1, 3, HEIGHT, WIDTH], "output_shape": list(exported.shape), "color": "RGB", "normalization": "ImageNet mean/std; see segmentation.py", "max_abs_logit_error": max_abs_error, "cpu_latency_median_ms": float(np.median(times)), "checkpoint_epoch": checkpoint["epoch"], "source_fingerprint": checkpoint["fingerprint"], "artifact_bytes": artifact.stat().st_size}
    (OUT / "export.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
