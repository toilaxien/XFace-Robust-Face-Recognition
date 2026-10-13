import torch

from xface.losses.adaptive_margin import RunningStandardizer, adaptive_angular_margin, ramp_factor


def test_adaptive_margin_is_bounded():
    zq = torch.tensor([-10.0, 0.0, 10.0])
    zg = torch.tensor([10.0, 0.0, -10.0])
    margin = adaptive_angular_margin(zq, zg, min_margin=0.34, max_margin=0.46)
    assert torch.all(margin >= 0.34)
    assert torch.all(margin <= 0.46)


def test_running_standardizer_shape():
    standardizer = RunningStandardizer()
    values = torch.randn(8)
    z = standardizer(values)
    assert z.shape == values.shape


def test_ramp_factor():
    assert ramp_factor(0, warmup=8, ramp=25) == 0.0
    assert ramp_factor(40, warmup=8, ramp=25) == 1.0
