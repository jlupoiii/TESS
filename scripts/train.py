#!/usr/bin/env python
"""
BEAM: Background Elimination with Advanced Machine learning - Training Script

This script trains a conditional diffusion model on TESS (Transiting Exoplanet 
Survey Satellite) Full Frame Images.
"""

import os
import argparse
import torch
import torch.multiprocessing as mp

from beam.data.datasets import TESSDataset, create_train_valid_datasets
from beam.training.trainer import DiffusionTrainer
from beam.training.distributed import run_distributed
from beam.utils.config import load_config, prepare_training_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a diffusion model on TESS images")
    
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def prepare_dataset(config):
    """
    Prepare the TESS dataset for training and validation.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (training dataset, validation dataset)
    """
    # Create dataset
    print(f"Creating TESSDataset using {config['data_num_processes']} processes...")
    tess_dataset = TESSDataset(
        angle_filename=config['data_angle_filename'],
        ccd_folder=config['data_ccd_folder'],
        image_shape=config['data_image_shape'],
        num_processes=config['data_num_processes']
    )
    
    # Create training and validation splits based on orbit number
    print("Creating training and validation splits...")
    train_split_criteria = lambda x: int(x['orbit']) <= 47
    valid_split_criteria = lambda x: int(x['orbit']) > 47
    
    train_dataset, valid_dataset = create_train_valid_datasets(
        tess_dataset, 
        train_split_criteria, 
        valid_split_criteria
    )
    
    # Print dataset statistics
    print(f'Full dataset has {len(tess_dataset)} datapoints')
    print(f'Training dataset has {len(train_dataset)} datapoints')
    print(f'Validation dataset has {len(valid_dataset)} datapoints')
    
    return train_dataset, valid_dataset


def train_worker(rank, world_size, config, train_dataset, valid_dataset, resume_path=None, device=None):
    """
    Worker function for training on a single GPU.
    
    Args:
        rank: GPU rank
        world_size: Total number of GPUs
        config: Training configuration
        train_dataset: Training dataset (pre-created in main process)
        valid_dataset: Validation dataset (pre-created in main process)
        resume_path: Path to checkpoint to resume from
        device: Device to train on
    """
    # Set device if not provided
    if device is None:
        device = torch.device(f'cuda:{rank}')
    
    print(f"GPU {rank}: Starting worker with {len(train_dataset)} training samples "
          f"and {len(valid_dataset)} validation samples")
    
    # Create trainer
    trainer = DiffusionTrainer(
        config=config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        rank=rank,
        world_size=world_size,
        device=device
    )
    
    # Resume from checkpoint if specified
    if resume_path:
        trainer.load_checkpoint(resume_path)
        print(f"GPU {rank}: Resumed training from {resume_path}")
    
    # Train the model
    trainer.train(
        checkpoint_freq=config['checkpoint_epoch_checkpoint'],
        save_model=config['checkpoint_save_model']
    )
    
    print(f"GPU {rank}: Training complete")


def main():
    """Main function to set up and launch distributed training."""
    # Parse arguments
    args = parse_args()
    
    # Load and process configuration
    config = load_config(args.config)
    flat_config = prepare_training_config(config)
    
    # Get world size
    world_size = flat_config['distributed_world_size']
    print(f"Using {world_size} GPUs for training")
    
    # Prepare dataset once in the main process before spawning workers
    print("Preparing dataset in main process...")
    train_dataset, valid_dataset = prepare_dataset(flat_config)
    
    # Run distributed training with pre-created datasets
    run_distributed(
        train_worker,
        world_size=world_size,
        args=(flat_config, train_dataset, valid_dataset, args.resume),
    )
    
    print("Training complete!")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()