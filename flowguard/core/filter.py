import torch
from typing import Protocol, Optional, Tuple
from .scoring import StepResidualScorer


class VelocityModel(Protocol):
    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ...


class FilterStats:
    def __init__(self):
        self.total_processed = 0
        self.total_filtered = 0
    
    def update(self, processed: int, filtered: int):
        self.total_processed += processed
        self.total_filtered += filtered
    
    def get_filter_percentage(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return (self.total_filtered / self.total_processed) * 100.0
    
    def reset(self):
        self.total_processed = 0
        self.total_filtered = 0


class FlowGuardFilter:
    def __init__(
        self, 
        model: VelocityModel,
        tau: float,
        scorer: Optional[StepResidualScorer] = None
    ):
        self.model = model
        self.tau = tau
        self.scorer = scorer or StepResidualScorer()
        self.stats = FilterStats()
    
    def generate_filtered(
        self,
        batch_size: int,
        data_shape: tuple,
        n_steps: int = 30,
        device: str = "cuda",
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x = torch.randn(batch_size, *data_shape, device=device)
            dt = 1.0 / n_steps
            t_0 = 0.0
            x_0 = x
            v_0 = self.model(torch.full((x_0.shape[0],), t_0, device=device), x_0)
            
            active_mask = torch.ones(x_0.shape[0], dtype=torch.bool, device=device)
            initial_batch_size = x_0.shape[0]
            
            for step in range(n_steps):
                if not active_mask.any():
                    break
                
                x_1 = x_0 + v_0 * dt
                current_time = dt * (step + 1)
                
                v_1_active = self.model(
                    torch.full((active_mask.sum(),), current_time, device=device),
                    x_1[active_mask]
                )
                
                v_1 = v_0.clone()
                v_1[active_mask] = v_1_active
                
                s_k = self.scorer.compute_score(v_0, v_1, dt, current_time)
                active_mask = active_mask & ~(s_k > self.tau)
                
                if verbose and step % 10 == 0:
                    print(f"Step {step}: {active_mask.sum().item()}/{x_0.shape[0]} samples active")
                
                x_0 = x_1
                v_0 = v_1
            
            filtered_count = initial_batch_size - active_mask.sum().item()
            self.stats.update(initial_batch_size, filtered_count)
            
            if verbose:
                print(f"Final: {active_mask.sum().item()}/{initial_batch_size} samples active")
            
            return x_0[active_mask], active_mask
    
    def get_stats(self) -> FilterStats:
        return self.stats
    
    def reset_stats(self):
        self.stats.reset()