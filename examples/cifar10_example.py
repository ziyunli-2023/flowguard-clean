import torch
import numpy as np
from flowguard import FlowGuardFilter, Calibrator
from flowguard.models import ModelWrapper
from flowguard.utils import FlowGuardConfig


def main():
    config = FlowGuardConfig(
        alpha=0.1,
        n_steps=30,
        batch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Example: Load your model (replace with actual model loading)
    # model = ModelWrapper.from_checkpoint(
    #     model_class=YourModelClass,
    #     checkpoint_path="path/to/checkpoint.pt",
    #     device=config.device,
    #     # model parameters...
    # )
    
    # For demonstration, create a dummy model
    class DummyModel:
        def __call__(self, t, x):
            return torch.randn_like(x) * 0.1
    
    model = DummyModel()
    
    # Calibration
    calibrator = Calibrator()
    
    # Option 1: Calibrate from scratch
    tau = calibrator.calibrate(
        model=model,
        data_shape=(3, 32, 32),  # CIFAR-10 shape
        device=config.device,
        n_cal=config.calibration_samples,
        alpha=config.alpha,
        n_steps=config.n_steps
    )
    
    # Option 2: Load pre-computed scores
    # tau = calibrator.load_scores_and_find_tau("scores.npy", config.alpha)
    
    # Create filter
    filter = FlowGuardFilter(model=model, tau=tau)
    
    # Generate filtered samples
    filtered_samples, active_mask = filter.generate_filtered(
        batch_size=config.batch_size,
        data_shape=(3, 32, 32),
        n_steps=config.n_steps,
        device=config.device,
        verbose=config.verbose
    )
    
    # Get statistics
    stats = filter.get_stats()
    print(f"Filter statistics:")
    print(f"  Total processed: {stats.total_processed}")
    print(f"  Total filtered: {stats.total_filtered}")
    print(f"  Filter percentage: {stats.get_filter_percentage():.2f}%")
    
    # Convert to images (for CIFAR-10)
    images = (filtered_samples * 127.5 + 128).clip(0, 255).to(torch.uint8)
    print(f"Generated {len(images)} filtered images")


if __name__ == "__main__":
    main()