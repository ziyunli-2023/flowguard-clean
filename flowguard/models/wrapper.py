import torch
from typing import Any


class ModelWrapper:
    def __init__(self, model: Any):
        self.model = model
        self.model.eval()
    
    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.model(t, x)
    
    @classmethod
    def from_checkpoint(
        cls, 
        model_class: Any, 
        checkpoint_path: str, 
        device: str = "cuda",
        **model_kwargs
    ) -> "ModelWrapper":
        model = model_class(**model_kwargs).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("ema_model", checkpoint)
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
        
        return cls(model)