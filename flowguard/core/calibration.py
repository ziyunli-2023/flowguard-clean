import math
import numpy as np
import torch
from tqdm import tqdm
from typing import Protocol, Optional
from .scoring import StepResidualScorer


class VelocityModel(Protocol):
    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ...


class Calibrator:
    def __init__(self, scorer: Optional[StepResidualScorer] = None):
        self.scorer = scorer or StepResidualScorer()
    
    def calibrate(
        self,
        model: VelocityModel,
        data_shape: tuple,
        device: str = "cuda",
        n_cal: int = 2000,
        alpha: float = 0.1,
        n_steps: int = 30,
        solver: str = "euler"
    ) -> float:
        scores = []
        
        for _ in tqdm(range(n_cal), desc=f"Calibrating (solver: {solver})"):
            z0 = torch.randn(1, *data_shape, device=device)
            _, s = self.scorer.score_path(model, z0, n_steps, solver)
            scores.append(s)
        
        tau = self.find_tau(scores, alpha)
        print(f"[Calibrate] α={alpha:.2f}  τ={tau:.5g} (solver: {solver})")
        
        return tau
    
    @staticmethod
    def find_tau(scores: list[float], alpha: float = 0.1) -> float:
        k = math.ceil((1 - alpha) * (len(scores) + 1))
        tau = np.partition(scores, k - 1)[k - 1]
        return tau
    
    def load_scores_and_find_tau(
        self, 
        scores_path: str, 
        alpha: float = 0.1
    ) -> float:
        scores = np.load(scores_path)
        return self.find_tau(scores.tolist(), alpha)