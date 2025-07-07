import os
import sys
import torch
import numpy as np
from datetime import datetime
from flowguard import FlowGuardFilter, Calibrator, FIDEvaluator
from flowguard.models import ModelWrapper
from flowguard.utils import FlowGuardConfig


def setup_logging(output_dir: str, alpha: float, integration_steps: int):
    log_filename = f"log_alpha_{alpha}_integration_steps_{integration_steps}.txt"
    log_file = f"{output_dir}/{log_filename}"
    os.makedirs(output_dir, exist_ok=True)
    
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        with open(log_file, 'a', encoding='utf-8') as f:
            message = ' '.join(str(arg) for arg in args)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    
    return log_print


def main():
    # Configuration matching your original setup
    config = FlowGuardConfig(
        alpha=0.0,  # Set to 0.0 to match your original code
        n_steps=30,
        batch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True
    )
    
    # Original paths from your code
    input_dir = "examples/images/models/cifar"
    model_type = "otcfm"
    step = 400000
    num_gen = 50000
    output_dir = "examples/images/cifar10/logs"
    
    # Setup logging
    log_print = setup_logging(output_dir, config.alpha, config.n_steps)
    
    # Adjust num_gen based on alpha (from your original code)
    adjusted_num_gen = int(num_gen / (1.0 - config.alpha)) if config.alpha != 0 else num_gen
    
    # Load model using your original UNet setup
    try:
        from torchcfm.models.unet.unet import UNetModelWrapper
        
        model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_res_blocks=2,
            num_channels=128,  # Your original num_channel
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(config.device)
        
        # Load checkpoint
        checkpoint_path = f"{input_dir}/{model_type}_cifar10_weights_step_{step}.pt"
        log_print(f"Loading model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        state_dict = checkpoint["ema_model"]
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
        
        model.eval()
        log_print("Model loaded successfully")
        
    except ImportError:
        log_print("torchcfm not available, using dummy model for demonstration")
        class DummyModel:
            def __call__(self, t, x):
                return torch.randn_like(x) * 0.1
        model = DummyModel()
    
    # Load or compute tau
    scores_file = f"scores_cifar_local_euler_{config.n_steps}.npy"
    
    if os.path.exists(scores_file) and config.alpha != 0:
        log_print(f"Loading scores from {scores_file}")
        calibrator = Calibrator()
        tau = calibrator.load_scores_and_find_tau(scores_file, config.alpha)
    elif config.alpha != 0:
        log_print("Calibrating threshold...")
        calibrator = Calibrator()
        tau = calibrator.calibrate(
            model=model,
            data_shape=(3, 32, 32),
            device=config.device,
            n_cal=2000,
            alpha=config.alpha,
            n_steps=config.n_steps
        )
        # Save scores for future use
        # Note: You'd need to modify Calibrator to return scores for saving
    else:
        tau = float('inf')  # No filtering when alpha=0
    
    log_print(f"alpha: {config.alpha}, tau: {tau}")
    
    # Create FlowGuard filter
    filter = FlowGuardFilter(model=model, tau=tau)
    
    # Create FID evaluator
    fid_evaluator = FIDEvaluator(
        filter=filter,
        data_shape=(3, 32, 32),
        dataset_name="cifar10",
        dataset_split="train",
        mode="legacy_tensorflow"
    )
    
    # Test generation first
    log_print("Testing filtered generation...")
    test_samples, test_mask = filter.generate_filtered(
        batch_size=10,
        data_shape=(3, 32, 32),
        n_steps=config.n_steps,
        device=config.device,
        verbose=True
    )
    log_print(f"Test generation successful: {len(test_samples)} samples generated")
    
    # Compute FID with filtering
    log_print("Start computing FID with filtering")
    
    result = fid_evaluator.compute_fid_with_stats(
        num_gen=adjusted_num_gen,
        batch_size=config.batch_size,
        n_steps=config.n_steps,
        device=config.device,
        dataset_res=32,
        verbose=True
    )
    
    # Log results
    log_print()
    log_print("FID computation completed")
    log_print(f"Model path: {checkpoint_path}")
    log_print(f"{config.n_steps} steps Euler FID with filtering (tau={tau}): {result['fid_score']:.4f}")
    log_print(f"Filter statistics:")
    log_print(f"  Total samples processed: {result['total_processed']}")
    log_print(f"  Total samples filtered: {result['total_filtered']}")
    log_print(f"  Filter percentage: {result['filter_percentage']:.2f}%")


if __name__ == "__main__":
    main()