import torch

from xface.losses.reliability_weighting import reliability_weights


def test_reliability_weights_are_bounded():
    zq = torch.tensor([2.0, -2.0, 0.5])
    zg = torch.tensor([-2.0, 2.0, 0.5])
    weights = reliability_weights(zq, zg, max_down_weight=0.10)
    assert torch.all(weights >= 0.90)
    assert torch.all(weights <= 1.00)
