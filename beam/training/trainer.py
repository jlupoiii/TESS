"""
Training functionality for diffusion models.

This module contains classes and functions for training and evaluating
diffusion models on TESS data.
"""

import os
import time
import pickle
from typing import Dict, List, Tuple, Optional, Union, Callable
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from beam.models.unet import ContextUnet
from beam.models.diffusion import DDPM
from beam.utils.visualization import plot_samples, plot_loss_history


class DiffusionTrainer:
    """
    Trainer class for diffusion models.
    """
    def __init__(
        self,
        config: Dict,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Create data loaders
        self.train_loader, self.valid_loader = self._create_data_loaders()
        
        # Create model
        self.model = self._create_model()
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['training_lrate']
        )
        
        # Initialize training state
        self.loss_history_train = []
        self.loss_history_valid = []
        self.time_history = []
        self.best_valid_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.start_training_time = None
        
        # Create checkpoint directory
        self.save_dir = config['paths_save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save dataset info
        if rank == 0:
            self._save_dataset_splits()
            self._save_model_info()
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training and validation.
        
        Returns:
            Tuple of (training data loader, validation data loader)
        """
        # Create distributed samplers if using multiple GPUs
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                self.train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank, 
                shuffle=True
            )
            
            valid_sampler = DistributedSampler(
                self.valid_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank, 
                shuffle=False
            )
        else:
            train_sampler = None
            valid_sampler = None
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config['training_batch_size'], 
            pin_memory=False,
            num_workers=0, 
            drop_last=True, 
            sampler=train_sampler,
            shuffle=(train_sampler is None)
        )
        
        valid_loader = DataLoader(
            self.valid_dataset, 
            batch_size=self.config['training_batch_size'], 
            pin_memory=False,
            num_workers=0, 
            drop_last=True, 
            sampler=valid_sampler,
            shuffle=False
        )
        
        return train_loader, valid_loader
    
    def _create_model(self) -> DDPM:
        """
        Create and initialize diffusion model.
        
        Returns:
            Initialized DDPM model
        """
        # Get conditioning dimension from dataset
        sample = next(iter(self.train_loader))
        in_dim = sample['x'].shape[2]  # Dimension of the conditioning vector
        
        # Create U-Net model
        unet = ContextUnet(
            in_channels=1, 
            in_dim=in_dim, 
            n_feat=self.config['model_n_feat']
        )
        
        # Create DDPM model
        ddpm = DDPM(
            nn_model=unet, 
            betas=(1e-4, 0.02), 
            n_T=self.config['model_n_T'], 
            device=self.device, 
            drop_prob=0.1
        )
        
        # Wrap model for distributed training if using multiple GPUs
        if self.world_size > 1:
            ddpm = nn.parallel.DistributedDataParallel(
                ddpm, 
                device_ids=[self.rank]
            )
        
        return ddpm
    
    def _save_dataset_splits(self) -> None:
        """
        Save information about training and validation dataset splits.
        """
        # Only save if this is the main process
        if self.rank != 0:
            return
            
        # Helper function to extract FFI numbers from a dataset
        def get_ffi_nums(dataset):
            if isinstance(dataset, Subset):
                ffi_nums = []
                for idx in dataset.indices:
                    sample = dataset.dataset[idx]
                    ffi_nums.append(sample['ffi_num'])
                return ffi_nums
            else:
                return [dataset[idx]['ffi_num'] for idx in range(len(dataset))]
        
        # Extract FFI numbers for each split
        training_dataset_ffis = get_ffi_nums(self.train_dataset)
        validation_dataset_ffis = get_ffi_nums(self.valid_dataset)
        
        # Save to pickle files
        with open(os.path.join(self.save_dir, 'training_dataset_ffinumbers.pkl'), 'wb') as file:
            pickle.dump(training_dataset_ffis, file)
            print(f"Saved training dataset FFI numbers to {file.name}")
            
        with open(os.path.join(self.save_dir, 'validation_dataset_ffinumbers.pkl'), 'wb') as file:
            pickle.dump(validation_dataset_ffis, file)
            print(f"Saved validation dataset FFI numbers to {file.name}")
    
    def _save_model_info(self) -> None:
        """
        Save model configuration information to a text file.
        """
        # Only save if this is the main process
        if self.rank != 0:
            return
            
        with open(os.path.join(self.save_dir, 'model_info.txt'), 'w') as f:
            f.write("MODEL AND DATA INFORMATION\n")
            f.write(f"Max number of epochs\t\t\t{self.config['training_n_epoch']}\n")
            f.write(f"Batch size\t\t\t\t\t\t{self.config['training_batch_size']}\n")
            f.write(f"Training/validation ratio\t\t{len(self.train_dataset) / (len(self.train_dataset) + len(self.valid_dataset)):.2f}\n")
            f.write(f"n_T (diffusion timesteps)\t\t{self.config['model_n_T']}\n")
            f.write(f"n_feat (CNN num of features)\t{self.config['model_n_feat']}\n")
            f.write(f"Learning rate\t\t\t\t\t{self.config['training_lrate']}\n")
            f.write(f"Image size\t\t\t\t\t\t{self.config['data_image_shape']}\n")
            f.write(f"GPUs used\t\t\t\t\t\t{self.world_size}\n")
            f.write(f"Early stopping patience\t\t{self.config['training_patience']}\n")
        
        print(f"Model info saved to {os.path.join(self.save_dir, 'model_info.txt')}")
    
    def save_checkpoint(self, epoch: int) -> None:
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
        """
        # Only save if this is the main process
        if self.rank != 0:
            return
            
        # Get base model (remove DistributedDataParallel wrapper if needed)
        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.save_dir, f"model_epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.loss_history_train,
            'valid_loss': self.loss_history_valid,
            'best_valid_loss': self.best_valid_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
        }, checkpoint_path)
        
        print(f'Saved checkpoint to {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get base model (remove DistributedDataParallel wrapper if needed)
        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        
        # Load model and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.loss_history_train = checkpoint['train_loss']
        self.loss_history_valid = checkpoint['valid_loss']
        self.best_valid_loss = checkpoint['best_valid_loss']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']
        
        print(f'Loaded checkpoint from {checkpoint_path} (epoch {checkpoint["epoch"]})')
    
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        # Set to training mode
        self.model.train()
        
        # Set epoch for distributed sampling
        if self.world_size > 1 and isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        # Apply learning rate decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config['training_lrate'] * (
                1 - self.current_epoch / self.config['training_n_epoch']
            )
        
        # Training loop
        loss_ema_train = None
        
        for data_batch in tqdm(self.train_loader, desc=f"Training epoch {self.current_epoch}"):
            self.optimizer.zero_grad()
            
            # Move data to device
            x_train = data_batch['y'].to(self.device)
            c_train = data_batch['x'].to(self.device)
            
            # Forward pass and loss calculation
            loss_train = self.model(x_train, c_train)
            
            # Backward pass
            loss_train.backward()
            self.optimizer.step()
            
            # Update exponential moving average of loss
            if loss_ema_train is None:
                loss_ema_train = loss_train.item()
            else:
                loss_ema_train = 0.95 * loss_ema_train + 0.05 * loss_train.item()
        
        return loss_ema_train
    
    def validate_epoch(self) -> float:
        """
        Validate the model on the validation dataset.
        
        Returns:
            Average validation loss
        """
        # Set to evaluation mode
        self.model.eval()
        
        # Validation loop
        loss_ema_valid = None
        
        with torch.no_grad():
            for data_batch in tqdm(self.valid_loader, desc=f"Validation epoch {self.current_epoch}"):
                # Move data to device
                x_valid = data_batch['y'].to(self.device)
                c_valid = data_batch['x'].to(self.device)
                
                # Forward pass and loss calculation
                loss_valid = self.model(x_valid, c_valid)
                
                # Update exponential moving average of loss
                if loss_ema_valid is None:
                    loss_ema_valid = loss_valid.item()
                else:
                    loss_ema_valid = 0.95 * loss_ema_valid + 0.05 * loss_valid.item()
        
        return loss_ema_valid
    
    def run_checkpoint(self, save_model: bool) -> bool:
        """
        Run checkpoint operations (saving model, generating samples, plotting)
        
        Args:
            save_model: Whether to save the model weights
            
        Returns:
            True if early stopping should be triggered, False otherwise
        """
        # Only run on the designated GPU
        if self.rank != self.config['checkpoint_checkpoint_gpu']:
            return False
            
        print(f'Running checkpoint at epoch {self.current_epoch} on GPU {self.rank}')
        
        try:
            # Get base model (remove DistributedDataParallel wrapper if needed)
            model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            
            # Generate and save sample images
            image_shape = self.config['data_image_shape']
            
            # Plot training samples
            plot_samples(
                model=model,
                dataloader=self.train_loader,
                device=self.device,
                n_sample=3,
                n_datapoint=3,
                image_shape=image_shape,
                title=f"Training Samples (Epoch {self.current_epoch})",
                save_path=os.path.join(self.save_dir, f"image_ep{self.current_epoch}_train.pdf")
            )
            
            # Plot validation samples
            plot_samples(
                model=model,
                dataloader=self.valid_loader,
                device=self.device,
                n_sample=5,
                n_datapoint=5,
                image_shape=image_shape,
                title=f"Validation Samples (Epoch {self.current_epoch})",
                save_path=os.path.join(self.save_dir, f"image_ep{self.current_epoch}_valid.pdf")
            )
            
            # Save loss plots
            plot_loss_history(
                train_losses=self.loss_history_train,
                valid_losses=self.loss_history_valid,
                save_dir=self.save_dir,
                show_last_n=50
            )
            
            # Save loss history to text file
            self._save_loss_history()
            
            # Save model
            if save_model:
                self.save_checkpoint(self.current_epoch)
            
            # Check for early stopping
            early_stop = (self.epochs_without_improvement >= self.config['training_patience'] and 
                        self.current_epoch >= self.config['training_patience'])
            
            return early_stop
            
        except Exception as e:
            print(f"Error in checkpoint on GPU {self.rank}: {e}")
            traceback.print_exc()
            # Don't stop training just because checkpoint failed
            return False
    
    def _save_loss_history(self) -> None:
        """
        Save loss history to a text file.
        """
        with open(os.path.join(self.save_dir, 'loss_history.txt'), 'w') as f:
            f.write("Epoch\tTraining MSE Loss\tValidation MSE Loss\tTime(Hrs)\n")
            
            for i in range(len(self.loss_history_valid)):
                f.write(
                    f"{i}\t\t{self.loss_history_train[i]:.4e}\t\t"
                    f"{self.loss_history_valid[i]:.4e}\t\t{self.time_history[i]}\n"
                )
            
        print(f"Loss history saved to {os.path.join(self.save_dir, 'loss_history.txt')}")
    
    def train(
        self, 
        n_epoch: Optional[int] = None, 
        checkpoint_freq: int = 10,
        save_model: bool = False
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            n_epoch: Number of epochs to train for (None = use config value)
            checkpoint_freq: Run checkpoint operations every N epochs
            save_model: Whether to save model weights at checkpoints
            
        Returns:
            Tuple of (training loss history, validation loss history)
        """
        # Set number of epochs
        n_epoch = n_epoch or self.config['training_n_epoch']
        
        # Initialize timing
        self.start_training_time = time.time()
        
        # Training loop
        for epoch in range(self.current_epoch, n_epoch):
            self.current_epoch = epoch
            print(f'Epoch {epoch}/{n_epoch}, GPU {self.rank}')
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.loss_history_train.append(train_loss)
            
            # Validate
            valid_loss = self.validate_epoch()
            self.loss_history_valid.append(valid_loss)

            # Memory cleanup here
            torch.cuda.empty_cache()
            
            # Update time history
            current_time = round((time.time() - self.start_training_time) / 3600, 3)  # Hours
            self.time_history.append(current_time)
            
            # Check for improvement
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Print status
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Valid Loss = {valid_loss:.6f}, "
                  f"Time = {current_time:.2f} hours")
            
            # Run checkpoint operations
            run_checkpoint = (epoch % checkpoint_freq == 0 or 
                             epoch == n_epoch - 1 or 
                             self.epochs_without_improvement >= self.config['training_patience'])
            
            if run_checkpoint:
                early_stop = self.run_checkpoint(save_model)
                
                # Synchronize early stopping across all GPUs
                if self.world_size > 1:
                    early_stop_tensor = torch.tensor([1 if early_stop else 0], device=self.device)
                    dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.MAX)
                    early_stop = early_stop_tensor.item() == 1
                
                if early_stop:
                    print(f"Early stopping triggered after {epoch} epochs on GPU {self.rank}")
                    break
        
        return self.loss_history_train, self.loss_history_valid
