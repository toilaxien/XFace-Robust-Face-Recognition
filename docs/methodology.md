# Methodology

XFace is a quality-adaptive and reliability-aware training strategy for face representation learning.

The method uses two training-time signals:

- Feature magnitude as a proxy for representation quality.
- Cosine gap between the target class and the strongest competing class as a class-separation signal.

Both signals are standardized using exponential moving average statistics. The standardized values are passed through bounded nonlinearities and used to compute a sample-specific additive angular margin.

The reliability weighting component reduces the influence of samples whose quality and cosine-gap signals conflict. This is designed as a conservative correction rather than a hard sample-removal rule.

At inference time, XFace uses the embedding network only. The adaptive margin, classifier head, running statistics and sample weighting are not needed for deployment.
