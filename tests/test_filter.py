import torch
import pytest
from flowguard import FlowGuardFilter, StepResidualScorer


class MockModel:
    def __call__(self, t, x):
        return torch.randn_like(x) * 0.1


def test_step_residual_scorer():
    scorer = StepResidualScorer()
    v1 = torch.randn(2, 3, 32, 32)
    v2 = torch.randn(2, 3, 32, 32)
    
    score = scorer.compute_score(v1, v2, dt=0.1, t_current=0.5)
    
    assert score.shape == (2,)
    assert torch.all(score >= 0)


def test_flowguard_filter():
    model = MockModel()
    tau = 1.0
    filter = FlowGuardFilter(model=model, tau=tau)
    
    filtered_samples, mask = filter.generate_filtered(
        batch_size=4,
        data_shape=(3, 8, 8),
        n_steps=5,
        device="cpu"
    )
    
    assert filtered_samples.shape[1:] == (3, 8, 8)
    assert len(filtered_samples) <= 4
    assert mask.dtype == torch.bool


def test_filter_stats():
    model = MockModel()
    filter = FlowGuardFilter(model=model, tau=0.0)  # Filter everything
    
    filter.generate_filtered(
        batch_size=10,
        data_shape=(3, 8, 8),
        n_steps=5,
        device="cpu"
    )
    
    stats = filter.get_stats()
    assert stats.total_processed == 10
    assert stats.get_filter_percentage() >= 0