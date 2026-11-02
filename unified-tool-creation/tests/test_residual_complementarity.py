import torch
from torch.nn import functional as F

from tools_builder.residual_complementarity import ResidualComplementarity


def test_residual_model_starts_as_exact_normalized_identity():
    torch.manual_seed(42)
    values = torch.randn(12, 16)
    expected = F.normalize(values, dim=-1)
    model = ResidualComplementarity(16, rank=4, alpha=4)
    for training in (False, True):
        model.train(training)
        assert torch.equal(model.project_anchor(values), expected)
        assert torch.equal(model.project_complement(values), expected)


def test_first_update_reaches_zero_initialized_b_not_a():
    torch.manual_seed(42)
    values = torch.randn(4, 8)
    model = ResidualComplementarity(8, rank=2, alpha=2)
    loss = -(model.project_anchor(values) * torch.roll(values, 1, 0)).sum()
    loss.backward()
    assert model.anchor_head.b.weight.grad.abs().sum() > 0
    assert model.anchor_head.a.weight.grad.abs().sum() == 0
