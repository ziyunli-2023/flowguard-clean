import torch
from typing import Protocol


class VelocityModel(Protocol):
    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ...


class StepResidualScorer:
    def __init__(self):
        pass
    
    def compute_score(
        self,
        v_current: torch.Tensor,
        v_next: torch.Tensor,
        dt: float,
        t_current: float
    ) -> torch.Tensor:
        residual = (v_current - v_next).pow(2)
        score = residual.mean(dim=(1, 2, 3)) * ((1 - t_current) ** 2 / dt)
        return score
    
    def score_path(
        self,
        model: VelocityModel,
        z0: torch.Tensor,
        n_steps: int = 100,
        solver: str = "euler"
    ) -> tuple[torch.Tensor, float]:
        device = z0.device
        dt = 1.0 / n_steps
        t_k = 0.0
        x_k = z0
        v_k = model(torch.full((x_k.shape[0],), t_k, device=device), x_k)
        s_max = 0.0
        
        for i in range(n_steps):
            x_k1 = x_k + v_k * dt
            t_k1 = t_k + dt
            
            v_k1 = model(torch.full((x_k1.shape[0],), t_k1, device=device), x_k1)
            
            s_k = self.compute_score(v_k, v_k1, dt, t_k1).item()
            s_max = max(s_max, s_k)
            
            x_k, v_k, t_k = x_k1, v_k1, t_k1
        
        return x_k, s_max