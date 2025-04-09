"""
Training callbacks for monitoring and controlling training.

This module provides callback classes that can be used to monitor and
control the training process.
"""

from typing import Dict, List, Optional, Callable, Any
import time
import os
import torch
import numpy as np


class Callback:
    """Base class for callbacks."""
    
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass
        
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of an epoch."""
        pass
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        """Called at the end of an epoch."""
        pass
        
    def on_batch_begin(self, trainer: Any, batch: int) -> None:
        """Called at the beginning of a batch."""
        pass
        
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict) -> None:
        """Called at the end of a batch."""
        pass


class EarlyStopping(Callback):
    """
    Callback to stop training when a monitored metric has stopped improving.
    
    Args:
        monitor: Metric to monitor
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' or 'max' (whether lower or higher values are better)
    """
    
    def __init__(
        self, 
        monitor: str = 'val_loss', 
        patience: int = 10, 
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.wait = 0
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.stopped_epoch = 0
        self.best_weights = None
        
    def on_train_begin(self, trainer: Any) -> None:
        """Reset wait counter and best value at the start of training."""
        self.wait = 0
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        """Check if training should be stopped at the end of the epoch."""
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.mode == 'min':
            improvement = (self.best - current) > self.min_delta
        else:
            improvement = (current - self.best) > self.min_delta
            
        if improvement:
            self.best = current
            self.wait = 0
            # Save model weights
            if hasattr(trainer, 'model'):
                model = trainer.model
                if hasattr(model, 'module'):
                    model = model.module
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                # Restore best weights
                if self.best_weights is not None and hasattr(trainer, 'model'):
                    model = trainer.model
                    if hasattr(model, 'module'):
                        model = model.module
                    model.load_state_dict(self.best_weights)


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints.
    
    Args:
        filepath: Path to save checkpoints to
        monitor: Metric to monitor
        save_best_only: Only save if the monitored metric improves
        save_weights_only: Only save model weights, not the whole model
        mode: 'min' or 'max' (whether lower or higher values are better)
        period: Save frequency (in epochs)
    """
    
    def __init__(
        self, 
        filepath: str, 
        monitor: str = 'val_loss', 
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = 'min',
        period: int = 1
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.period = period
        self.epochs_since_last_save = 0
        self.best = float('inf') if mode == 'min' else float('-inf')
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        """Save checkpoint at the end of the epoch if conditions are met."""
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            
            # Format filepath with epoch and metrics
            filepath = self.filepath.format(epoch=epoch, **logs)
            
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    return
                    
                if self.mode == 'min':
                    improvement = current < self.best
                else:
                    improvement = current > self.best
                    
                if improvement:
                    self.best = current
                    self._save_model(trainer, filepath)
            else:
                self._save_model(trainer, filepath)
    
    def _save_model(self, trainer: Any, filepath: str) -> None:
        """Save the model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.save_weights_only:
            # Save only weights
            model = trainer.model
            if hasattr(model, 'module'):
                model = model.module
            torch.save(model.state_dict(), filepath)
        else:
            # Save full checkpoint
            checkpoint = {
                'epoch': trainer.current_epoch,
                'model_state_dict': trainer.model.module.state_dict() 
                    if hasattr(trainer.model, 'module') 
                    else trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': trainer.loss_history_train,
                'valid_loss': trainer.loss_history_valid,
            }
            torch.save(checkpoint, filepath)


class LearningRateScheduler(Callback):
    """
    Callback to adjust learning rate according to a schedule.
    
    Args:
        schedule: Function to calculate learning rate given epoch
        verbose: Whether to print learning rate updates
    """
    
    def __init__(self, schedule: Callable[[int], float], verbose: int = 0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose
        
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Set learning rate at the beginning of each epoch."""
        if not hasattr(trainer, 'optimizer'):
            return
            
        lr = self.schedule(epoch)
        
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = lr
            
        if self.verbose > 0:
            print(f'\nEpoch {epoch}: setting learning rate to {lr:.6f}')


class TensorBoardLogger(Callback):
    """
    Callback to log metrics to TensorBoard.
    
    Args:
        log_dir: Directory to save TensorBoard logs
    """
    
    def __init__(self, log_dir: str = 'logs'):
        super().__init__()
        self.log_dir = log_dir
        
    def on_train_begin(self, trainer: Any) -> None:
        """Initialize TensorBoard writer at the start of training."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            print("Warning: TensorBoard not available, logging disabled")
            self.writer = None
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict) -> None:
        """Log metrics at the end of each epoch."""
        if self.writer is None:
            return
            
        for name, value in logs.items():
            self.writer.add_scalar(name, value, epoch)
            
        # Add learning rate
        if hasattr(trainer, 'optimizer'):
            lr = trainer.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('learning_rate', lr, epoch)
            
    def on_train_end(self, trainer: Any) -> None:
        """Close TensorBoard writer at the end of training."""
        if self.writer is not None:
            self.writer.close()
