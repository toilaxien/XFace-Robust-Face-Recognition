# XFace - Quality-Adaptive and Reliability-Aware Learning for Robust Face Recognition

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Face Recognition](https://img.shields.io/badge/Task-Face%20Recognition-0A66C2?style=flat)
![Robustness](https://img.shields.io/badge/Focus-Robustness-2E8B57?style=flat)
![CASIA-WebFace](https://img.shields.io/badge/Dataset-CASIA--WebFace-6A5ACD?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

**XFace** is a face representation learning framework for robust face recognition under degraded image quality. It combines **feature-quality modeling**, **cosine-gap class separation**, and **reliability-aware sample weighting** to adapt the angular margin during training.

The project compares XFace with strong margin-based and adaptive baselines, including **ArcFace**, **CosFace**, **AdaFace**, **MagFace**, and **HAMFace**, using a shared verification protocol on LFW, AgeDB-30, CALFW, and CPLFW.

> Academic research project on robust face recognition  
> Industrial University of Ho Chi Minh City - Faculty of Information Technology

---

## Abstract

Face recognition systems remain sensitive to blur, noise, low resolution, illumination change, and occlusion. XFace addresses this limitation by using the magnitude of the face embedding as a representation-quality proxy and the cosine gap between the target class and the strongest competing class as a separation signal. These two signals are standardized with running statistics and used to derive a sample-specific additive angular margin. A conservative reliability-aware weighting mechanism further reduces the contribution of samples with conflicting quality and separation signals.

All XFace-specific mechanisms are applied only during training. At inference time, the model uses the same backbone and embedding network as the compared baselines, adding no extra parameters or computational complexity.

## Repository Structure

```text
XFace-Robust-Face-Recognition/
|-- configs/                # Training and evaluation configuration files
|-- src/xface/              # Reusable XFace Python package
|-- scripts/                # Command-line entry points
|-- tests/                  # Unit tests for losses, weighting and embeddings
|-- results/                # Benchmark, robustness and ablation results
|-- docs/                   # Methodology, experiments and reproduction notes
|-- data/                   # Dataset instructions
`-- checkpoints/            # Checkpoint instructions
```

## Method Overview

Given an input face image, the backbone produces an unnormalized embedding. XFace uses:

- **Quality signal**: feature magnitude, used as a proxy for representation quality.
- **Cosine-gap signal**: target-class cosine similarity minus the strongest non-target similarity.
- **Adaptive angular margin**: sample-specific margin derived from standardized quality and cosine-gap signals.
- **Reliability weighting**: conservative down-weighting for samples whose quality and separation signals conflict.

The final training objective is a cross-entropy loss over margin-adjusted logits with optional reliability-aware sample weights.

## Main Results

Average verification performance across LFW, AgeDB-30, CALFW and CPLFW:

| Method | Accuracy (%) | AUC | EER (%) | TAR@FAR=1e-2 | TAR@FAR=1e-3 |
|---|---:|---:|---:|---:|---:|
| ArcFace | 91.908 | 0.9660 | 8.108 | 76.900 | 60.325 |
| CosFace | 91.679 | 0.9627 | 8.525 | 75.538 | 50.742 |
| AdaFace | 91.692 | 0.9620 | 8.725 | 71.467 | 51.133 |
| MagFace | 90.004 | 0.9523 | 10.175 | 66.567 | 42.492 |
| HAMFace | 87.371 | 0.9301 | 13.100 | 59.942 | 39.792 |
| XFace | 90.933 | 0.9604 | 9.000 | 70.958 | 50.417 |

Robustness under controlled degradations:

| Method | Clean | Blur | Noise | Low-Res. | Brightness | Occlusion | Robust Mean | Drop | Retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ArcFace | 91.908 | 86.421 | 88.942 | 82.788 | 91.675 | 86.004 | 87.166 | 4.742 | 94.745 |
| CosFace | 91.679 | 84.596 | 88.100 | 80.792 | 91.550 | 85.838 | 86.175 | 5.504 | 93.879 |
| AdaFace | 91.692 | 83.096 | 87.267 | 77.650 | 91.388 | 84.142 | 84.708 | 6.983 | 92.221 |
| MagFace | 90.004 | 81.688 | 81.738 | 77.858 | 89.458 | 82.392 | 82.627 | 7.377 | 91.601 |
| HAMFace | 87.371 | 80.133 | 82.063 | 76.938 | 87.250 | 81.571 | 81.591 | 5.780 | 93.245 |
| XFace | 90.933 | 86.000 | 88.000 | 83.325 | 90.688 | 85.254 | 86.653 | 4.280 | 95.213 |

XFace does not maximize clean-benchmark accuracy, but it obtains the smallest average degradation-induced accuracy drop and the highest retention ratio among the evaluated methods.

## Installation

```bash
git clone https://github.com/<your-name>/XFace-Robust-Face-Recognition.git
cd XFace-Robust-Face-Recognition
pip install -e .
```

## Quick Start

Train XFace:

```bash
python scripts/train.py --config configs/xface.yaml
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py --config configs/xface.yaml --checkpoint checkpoints/xface_best.pth
```

Run robustness evaluation:

```bash
python scripts/evaluate_robustness.py --config configs/xface.yaml --checkpoint checkpoints/xface_best.pth
```

Run ablation variants:

```bash
python scripts/run_ablation.py --config configs/ablation.yaml
```

## Notes

Large datasets and model checkpoints are intentionally excluded from Git. See `data/README.md` and `checkpoints/README.md` for the expected local layout.
