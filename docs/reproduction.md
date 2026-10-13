# Reproduction

1. Install dependencies:

```bash
pip install -e .
```

2. Prepare CASIA-WebFace under `data/CASIA-WebFace`.

3. Train XFace:

```bash
python scripts/train.py --config configs/xface.yaml
```

4. Evaluate clean verification:

```bash
python scripts/evaluate.py --config configs/xface.yaml --checkpoint checkpoints/xface_best.pth
```

5. Evaluate robustness:

```bash
python scripts/evaluate_robustness.py --config configs/xface.yaml --checkpoint checkpoints/xface_best.pth
```

The current results are based on single-run experiments. Multi-seed validation is recommended for future reproduction studies.
