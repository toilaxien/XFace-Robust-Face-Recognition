# Experiments

## Standard Verification

Models are evaluated on LFW, AgeDB-30, CALFW and CPLFW using pair-based verification. Each benchmark uses 6,000 verification pairs and 10-fold threshold selection.

## Robustness Evaluation

The same verification protocol is repeated under five controlled degradations:

| Condition | Setting |
|---|---|
| Gaussian blur | Radius = 2.0 |
| Gaussian noise | Sigma = 0.08 |
| Low resolution | 112 -> 28 -> 112 |
| Brightness | Factor = 0.50 |
| Occlusion | 50% width, 30% height |

Robustness is reported with degraded accuracy, robust mean, average accuracy drop and retention ratio.

## Ablation

The ablation compares cumulative XFace variants:

- Q-only: quality-adaptive margin.
- Q+G: quality plus cosine-gap adaptive margin.
- Full XFace: quality, cosine gap and reliability-aware weighting.
