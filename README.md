# FieldLens

An independent, interactive work sample for [Carbon Robotics' Deep Learning Engineer role](https://carbonrobotics.com/job-openings?gh_jid=4673637006). **[Open the demo](https://andrewgordienko.github.io/fieldlens-carbon-robotics/)** and start with **Explore the cases**. The page shows field images, model predictions, ground truth, probability maps, threshold behavior, and a crop-overlap stress test. It uses no Carbon Robotics data or model.

## The result

A ResNet-18 encoder and U-Net-style decoder was trained for soil, rice crop, and weed segmentation. It reached **0.664 weed IoU** on held-out images. A weed threshold of **0.900**, chosen on validation images to keep crop pixels called weed below 1%, produced **1.64% crop-to-weed errors** and **70.4% weed-pixel recall** on test images. This fails the *illustrative* 1% crop-risk gate. The test-set crop-risk 95% interval is bootstrapped by image and appears in the demo.

The overlap stress test pastes crop cutouts taken only from training images over 60 held-out weed components. At a requested 40% cover, mean **measured cover was about 30%**, visible-weed recall fell to about **59%**, and about **25% of the pasted crop pixels** were marked weed. These are segmentation measurements on synthetic composites. They do not estimate laser targeting accuracy or crop damage. Requested and measured overlap are both shown because thin rice plants cannot always reach the requested cover.

## Reproduce

Python 3.11+, `bsdtar`, and internet access are needed for the [CC BY 4.0 dataset](https://figshare.com/articles/dataset/rice_seedlings_and_weeds/7488830) and ImageNet encoder weights. PyTorch training takes longer on CPU. The published page is a static snapshot of the evaluated run; local Python runs retrain and rebuild it.

```bash
cd fieldlens-carbon-robotics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py
python train.py --epochs 20
python evaluate.py
python export.py
python build_site.py
python3 -m http.server 8003 --directory docs
```

Open `http://127.0.0.1:8003/`. To check pipeline invariants: `python -m unittest discover -s tests -v`. To run the exported TorchScript model on an image: `python infer.py --help`. Generated data, checkpoints, reports, and local predictions go in ignored `data/` and `outputs/`; `docs/` contains the inspected static report and its image views.

## Evaluation design

- The [source paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0215676) describes 28 original field images cut into eight adjacent tiles each. All eight tiles stay together in the same split: 18 originals for training, five for validation, five for test.
- The best checkpoint is selected by mean crop and weed IoU on validation images. The weed action-score threshold is also selected on validation. Test results do not choose either.
- Ground-truth crop and weed components come from pixel masks. Stress crops are sampled from training tiles. The benchmark reports both visible weed recall and model activation on the inserted crop pixels.
- The decision gate (at most 1% crop pixels called weed and at least 70% weed recall) is an example research criterion, **not a Carbon Robotics specification**. The chart lets a reviewer inspect other thresholds without rewriting the chosen result.
- `export.py` traces and freezes the network as TorchScript and checks logit parity. The exported artifact for the reported run had a maximum absolute logit difference of `1.91e-6` and median CPU inference of about `43 ms` for one 384 × 288 image on the development machine. Timing is a machine-specific reference, not a LaserWeeder latency claim.

## Scope and rights

Carbon says its LaserWeeder uses machine vision to identify crops and weeds before laser targeting. This project examines one perception failure suggested by that workflow. It has no meristem labels, actuation planner, laser outcomes, Carbon crop images, or Carbon model. The rice paddy domain differs from production fields, and the pasted-plant test has compositing artifacts. Before using a similar gate at work, I would test on representative field video, measure calibration by crop and lighting conditions, and connect candidate weed regions to the actual targeting policy and damage outcomes.

Field images and masks: Ma et al. (2019), [*Fully convolutional network for rice seedling and weed image segmentation at the seedling stage in paddy fields*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0215676), [Figshare dataset](https://figshare.com/articles/dataset/rice_seedlings_and_weeds/7488830), CC BY 4.0. Source tiles were resized and combined with model overlays in the demo. Code is MIT licensed.
