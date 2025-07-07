import torch
from cleanfid import fid
from typing import Callable, Optional
from ..core.filter import FlowGuardFilter


class FIDEvaluator:
    def __init__(
        self,
        filter: FlowGuardFilter,
        data_shape: tuple,
        dataset_name: str = "cifar10",
        dataset_split: str = "train",
        mode: str = "legacy_tensorflow"
    ):
        self.filter = filter
        self.data_shape = data_shape
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.mode = mode
    
    def create_generator(self, batch_size: int, n_steps: int, device: str = "cuda") -> Callable:
        def generator(unused_latent):
            filtered_samples, _ = self.filter.generate_filtered(
                batch_size=batch_size,
                data_shape=self.data_shape,
                n_steps=n_steps,
                device=device,
                verbose=False
            )
            # Convert to uint8 images for FID computation
            images = (filtered_samples * 127.5 + 128).clip(0, 255).to(torch.uint8)
            return images
        
        return generator
    
    def compute_fid(
        self,
        num_gen: int,
        batch_size: int,
        n_steps: int,
        device: str = "cuda",
        dataset_res: int = 32
    ) -> float:
        generator = self.create_generator(batch_size, n_steps, device)
        
        score = fid.compute_fid(
            gen=generator,
            dataset_name=self.dataset_name,
            batch_size=batch_size,
            dataset_res=dataset_res,
            num_gen=num_gen,
            dataset_split=self.dataset_split,
            mode=self.mode,
        )
        
        return score
    
    def compute_fid_with_stats(
        self,
        num_gen: int,
        batch_size: int,
        n_steps: int,
        device: str = "cuda",
        dataset_res: int = 32,
        verbose: bool = True
    ) -> dict:
        if verbose:
            print("Computing FID with filtering...")
        
        # Reset stats before computation
        self.filter.reset_stats()
        
        score = self.compute_fid(num_gen, batch_size, n_steps, device, dataset_res)
        
        # Get filter statistics
        stats = self.filter.get_stats()
        
        result = {
            "fid_score": score,
            "total_processed": stats.total_processed,
            "total_filtered": stats.total_filtered,
            "filter_percentage": stats.get_filter_percentage(),
            "tau": self.filter.tau,
            "n_steps": n_steps
        }
        
        if verbose:
            print(f"FID Score: {score:.4f}")
            print(f"Filter Statistics:")
            print(f"  Total processed: {stats.total_processed}")
            print(f"  Total filtered: {stats.total_filtered}")
            print(f"  Filter percentage: {stats.get_filter_percentage():.2f}%")
        
        return result