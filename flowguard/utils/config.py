from dataclasses import dataclass
from typing import Optional


@dataclass
class FlowGuardConfig:
    alpha: float = 0.1
    n_steps: int = 30
    solver: str = "euler"
    batch_size: int = 1024
    device: str = "cuda"
    verbose: bool = False
    
    calibration_samples: int = 2000
    tau: Optional[float] = None
    scores_path: Optional[str] = None
    
    def __post_init__(self):
        if self.device == "cuda" and not self._cuda_available():
            self.device = "cpu"
            print("CUDA not available, using CPU")
    
    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False