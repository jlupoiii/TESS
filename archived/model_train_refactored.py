"""
BEAM: Background Elimination with Advanced Machine learning - Training Script

This script trains a conditional diffusion model on TESS (Transiting Exoplanet 
Survey Satellite) Full Frame Images. The model generates TESS images conditioned
on orbital parameters.

The code is adapted from:
https://github.com/TeaPearce/Conditional_Diffusion_MNIST
which was originally modified from:
https://github.com/cloneofsimo/minDiffusion

Based on research from:
- DDPM: https://arxiv.org/abs/2006.11239
- Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598
- ImageGen: https://arxiv.org/abs/2205.11487
"""

import os
import time
import pickle
import argparse
import multiprocessing
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms

# Import model components from utils.py
from utils_refactored import (
    ContextUnet, 
    DDPM, 
    ResidualConvBlock, 
    UnetDown, 
    UnetUp, 
    EmbedFC, 
    ddpm_schedules
)


class TESSDataset(Dataset):
    """
    Dataset for TESS (Transiting Exoplanet Survey Satellite) images and their
    corresponding orbital parameters.
    """
    def __init__(
        self,
        angle_filename: str,
        ccd_folder: str,
        image_shape: Tuple[int, int],
        num_processes: int = 20
    ):
        start_time = time.time()
        
        # Define paths and parameters
        self.angle_folder = "/pdo/users/jlupoiii/TESS/data/angles/"
        self.ccd_folder = ccd_folder
        self.image_shape = image_shape
        
        # Initialize data containers
        self.data = []       # Orbital parameters
        self.labels = []     # Image data
        self.ffi_nums = []   # FFI identification numbers
        
        # Load orbital parameter dictionary
        self.angles_dic = pickle.load(open(os.path.join(self.angle_folder, angle_filename), "rb"))
        
        # Find all valid image files that have corresponding angle data
        files = [
            filename for filename in os.listdir(self.ccd_folder)
            if filename[18:18+8] in self.angles_dic.keys()
        ]
        
        # Process files in parallel
        print(f"Loading {len(files)} image files using {num_processes} processes...")
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(self.load_images_worker, files)
            
            # Process results and build dataset
            for x, y, ffi_num in tqdm(results, desc="Processing results"):
                if x is not None:
                    self.data.append(x)
                    self.labels.append(y)
                    self.ffi_nums.append(ffi_num)
        
        
        # Report dataset loading time and size
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Dataset built with {len(self.data)} samples in {total_time:.2f} seconds")

    def load_images_worker(self, filename: str) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[str]]:
        """
        Worker function for parallel processing of image files.
        
        Args:
            filename: Name of the image file to process
            
        Returns:
            Tuple of (orbital parameters, image data, FFI number)
        """
        # Skip files we don't want (non-camera 3 files or malformed names)
        if len(filename) < 40 or filename[27] != '3': 
            return None, None, None

        # Load image data
        try:
            image_arr = pickle.load(open(os.path.join(self.ccd_folder, filename), "rb"))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, None, None
        
        # Extract FFI number and get corresponding angles
        ffi_num = filename[18:18+8]
        try:
            angles = self.angles_dic[ffi_num]
        except KeyError:
            return None, None, None
            
        # Prepare orbital parameters (12 values)
        params = np.array([
            angles['1/ED'], angles['1/MD'], 
            angles['1/ED^2'], angles['1/MD^2'], 
            angles['Eel'], angles['Eaz'], 
            angles['Mel'], angles['Maz'], 
            angles['E3el'], angles['E3az'], 
            angles['M3el'], angles['M3az']
        ])
        
        # Convert to images for consistent processing
        x = Image.fromarray(params)
        y = Image.fromarray(image_arr.flatten())

        return x, y, ffi_num

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary with:
                - x: Orbital parameters tensor
                - y: Image tensor
                - ffi_num: FFI identification number
                - orbit: Orbit number
        """
        angles_image = self.data[idx]
        ffi_image = self.labels[idx]
        ffi_num = self.ffi_nums[idx]
        orbit = self.angles_dic[ffi_num]["orbit"]

        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda s: s.reshape(1, angles_image.size[1])  # Reshape to 1×N tensor
        ])
        
        target_transform = transforms.Compose([
            lambda s: np.array(s),
            lambda s: s.reshape(self.image_shape),  # Reshape to the target image size
            transforms.ToTensor()
        ])

        # Apply transformations
        angles_image = transform(angles_image)
        ffi_image = target_transform(ffi_image)

        return {
            "x": angles_image,      # Orbital parameters (1×12 vector)
            "y": ffi_image,         # Image (64×64 or other size)
            "ffi_num": ffi_num,     # FFI identification number
            "orbit": orbit          # Orbit number
        }


def setup_distributed(rank: int, world_size: int) -> None:
    """
    Setup for distributed training.
    
    Args:
        rank: Index of the current GPU
        world_size: Total number of GPUs
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def save_model_info(save_dir: str, config: Dict) -> None:
    """
    Save model configuration information to a text file.
    
    Args:
        save_dir: Directory to save the information
        config: Dictionary of model parameters
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'model_info.txt'), 'w') as f:
        f.write("MODEL AND DATA INFORMATION\n")
        f.write(f"Max number of epochs\t\t\t{config['n_epoch']}\n")
        f.write(f"Batch size\t\t\t\t\t\t{config['batch_size']}\n")
        f.write(f"Training/validation ratio\t\t{config['train_ratio']}\n")
        f.write(f"n_T (diffusion timesteps)\t\t{config['n_T']}\n")
        f.write(f"n_feat (CNN num of features)\t{config['n_feat']}\n")
        f.write(f"Learning rate\t\t\t\t\t{config['lrate']}\n")
        f.write(f"Image size\t\t\t\t\t\t{config['image_shape']}\n")
        f.write(f"GPUs used\t\t\t\t\t\t{torch.cuda.device_count()}\n")
        f.write(f"Early stopping patience\t\t{config['patience']}\n")
    
    print(f"Model info saved to {os.path.join(save_dir, 'model_info.txt')}")


def prepare_dataset(
    angle_filename: str, 
    ccd_folder: str, 
    image_shape: Tuple[int, int], 
    num_processes: int,
    train_ratio: float,
    save_dir: str
) -> Tuple[Subset, Subset]:
    """
    Prepare the TESS dataset for training and validation.
    
    Args:
        angle_filename: Filename of the orbital parameters
        ccd_folder: Folder containing the processed images
        image_shape: Shape of the images
        num_processes: Number of processes for parallel loading
        train_ratio: Ratio of training/validation split
        save_dir: Directory to save dataset metadata
        
    Returns:
        Tuple of (training dataset, validation dataset)
    """
    # Create dataset
    tess_dataset = TESSDataset(angle_filename, ccd_folder, image_shape, num_processes)
    
    # Separate data into training and validation sets by orbit
    # Training: orbits 11-46, Validation: orbits 47-54
    train_indices = [idx for idx, data_point in enumerate(tess_dataset) if int(data_point["orbit"]) <= 47]
    valid_indices = [idx for idx, data_point in enumerate(tess_dataset) if int(data_point["orbit"]) > 47]
    
    train_dataset = Subset(tess_dataset, train_indices)
    valid_dataset = Subset(tess_dataset, valid_indices)
    
    # Print dataset statistics
    num_train_samples = len(train_indices)
    num_valid_samples = len(valid_indices)
    
    print(f'Full dataset has {num_train_samples + num_valid_samples} datapoints')
    print(f'Training dataset has {num_train_samples} datapoints')
    print(f'Validation dataset has {num_valid_samples} datapoints')
    print(f"Orbital parameter shape: {tess_dataset[0]['x'].shape}")
    print(f"Image shape: {tess_dataset[0]['y'].shape}")
    
    # Save dataset splits for reproducibility
    save_dataset_splits(train_dataset, valid_dataset, save_dir)
    
    return train_dataset, valid_dataset


def save_dataset_splits(train_dataset: Subset, valid_dataset: Subset, save_dir: str) -> None:
    """
    Save information about training and validation dataset splits.
    
    Args:
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        save_dir: Directory to save the dataset metadata
    """
    # Extract FFI numbers for each split
    training_dataset_ffis = [train_dataset[index]['ffi_num'] for index in range(len(train_dataset))]
    validation_dataset_ffis = [valid_dataset[index]['ffi_num'] for index in range(len(valid_dataset))]
    
    # Save to pickle files
    with open(os.path.join(save_dir, 'training_dataset_ffinumbers.pkl'), 'wb') as file:
        pickle.dump(training_dataset_ffis, file)
        print(f"Saved training dataset FFI numbers to {file.name}")
        
    with open(os.path.join(save_dir, 'validation_dataset_ffinumbers.pkl'), 'wb') as file:
        pickle.dump(validation_dataset_ffis, file)
        print(f"Saved validation dataset FFI numbers to {file.name}")


def train(rank: int, world_size: int, config: Dict) -> None:
    """
    Main training function for distributed training.
    
    Args:
        rank: Index of the current GPU
        world_size: Total number of GPUs
        config: Dictionary of model parameters
    """
    # Setup distributed training
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Get datasets from shared memory (prepared in the main process)
    train_dataset = config['train_dataset']
    valid_dataset = config['valid_dataset']
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    
    valid_sampler = DistributedSampler(
        valid_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        pin_memory=True,
        num_workers=2, 
        drop_last=True, 
        sampler=train_sampler
    )
    
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=config['batch_size'], 
        pin_memory=True,
        num_workers=2, 
        drop_last=True, 
        sampler=valid_sampler
    )
    
    print(f"GPU {rank}: {len(train_dataloader)} training batches, {len(valid_dataloader)} validation batches")
    
    # Create model
    in_dim = next(iter(valid_dataloader))['x'].shape[2]  # Dimension of the conditioning vector
    
    # Initialize model
    ddpm = DDPM(
        nn_model=ContextUnet(
            in_channels=1, 
            in_dim=in_dim, 
            n_feat=config['n_feat']
        ), 
        betas=(1e-4, 0.02), 
        n_T=config['n_T'], 
        device=device, 
        drop_prob=0.1
    )
    
    # Wrap model for distributed training
    ddpm = nn.parallel.DistributedDataParallel(ddpm, device_ids=[rank])
    
    # Create optimizer
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=config['lrate'])
    
    # Initialize tracking variables
    loss_history_train = []
    loss_history_valid = []
    time_history = []
    start_training_time = time.time()
    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    
    # Training loop
    for epoch in range(config['n_epoch']):
        print(f'Epoch {epoch}/{config["n_epoch"]} training, GPU {rank}')
        
        # Set to training mode
        ddpm.train()
        train_sampler.set_epoch(epoch)  # For proper shuffling in distributed training
        
        # Linear learning rate decay
        optimizer.param_groups[0]['lr'] = config['lrate'] * (1 - epoch / config['n_epoch'])
        
        # Training phase
        loss_ema_train = None
        for data_batch in tqdm(train_dataloader, desc=f"Training epoch {epoch}"):
            optimizer.zero_grad()
            
            # Move data to device
            x_train = data_batch['y'].to(device)
            c_train = data_batch['x'].to(device)
            
            # Forward pass and loss calculation
            loss_train = ddpm(x_train, c_train)
            
            # Backward pass
            loss_train.backward()
            optimizer.step()
            
            # Update exponential moving average of loss
            if loss_ema_train is None:
                loss_ema_train = loss_train.item()
            else:
                loss_ema_train = 0.95 * loss_ema_train + 0.05 * loss_train.item()
        
        # Record training loss for this epoch
        loss_history_train.append(loss_ema_train)
        
        # Validation phase
        ddpm.eval()
        with torch.no_grad():
            print(f'Epoch {epoch}/{config["n_epoch"]} validation, GPU {rank}')
            
            loss_ema_valid = None
            for data_batch in tqdm(valid_dataloader, desc=f"Validation epoch {epoch}"):
                # Move data to device
                x_valid = data_batch['y'].to(device)
                c_valid = data_batch['x'].to(device)
                
                # Forward pass and loss calculation
                loss_valid = ddpm(x_valid, c_valid)
                
                # Update exponential moving average of loss
                if loss_ema_valid is None:
                    loss_ema_valid = loss_valid.item()
                else:
                    loss_ema_valid = 0.95 * loss_ema_valid + 0.05 * loss_valid.item()
            
            # Record validation loss for this epoch
            loss_history_valid.append(loss_ema_valid)
            
            # Record elapsed time
            current_time = round((time.time() - start_training_time) / 3600, 3)  # Hours
            time_history.append(current_time)
            
            # Check for improvement
            if loss_ema_valid < best_valid_loss:
                best_valid_loss = loss_ema_valid
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Early stopping check
            early_stop = (epochs_without_improvement >= config['patience'] and 
                          epoch >= config['patience'])
            
            # Only process checkpoints on the designated GPU
            if ((epoch % config['epoch_checkpoint'] == 0 or 
                 epoch == config['n_epoch'] - 1 or 
                 early_stop) and 
                rank == config['checkpoint_gpu']):
                
                print(f'Running checkpoint at epoch {epoch} on GPU {config["checkpoint_gpu"]}')
                
                # Generate and save sample images
                save_checkpoint_samples(
                    ddpm=ddpm.module,
                    train_dataloader=train_dataloader,
                    valid_dataloader=valid_dataloader,
                    device=device,
                    epoch=epoch,
                    save_dir=config['save_dir'],
                    image_shape=config['image_shape'],
                    n_sample=5  # Number of samples per conditioning vector
                )
                
                # Save loss plots
                save_loss_plots(
                    loss_history_train=loss_history_train,
                    loss_history_valid=loss_history_valid,
                    save_dir=config['save_dir']
                )
                
                # Save loss history to text file
                save_loss_history(
                    loss_history_train=loss_history_train,
                    loss_history_valid=loss_history_valid,
                    time_history=time_history,
                    save_dir=config['save_dir']
                )
                
                # Save model
                if config['save_model']:
                    torch.save(
                        ddpm.module.state_dict(), 
                        os.path.join(config['save_dir'], f"model_epoch{epoch}.pth")
                    )
                    print(f'Saved model at {os.path.join(config["save_dir"], f"model_epoch{epoch}.pth")}')
            
            # Synchronize early stopping across all GPUs
            early_stop_tensor = torch.tensor([1 if early_stop else 0], device=device)
            dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.MAX)
            
            if early_stop_tensor.item() == 1:
                print(f"Early stopping triggered after {epoch} epochs on GPU {rank}")
                break
    
    # Clean up
    dist.destroy_process_group()


def save_checkpoint_samples(
    ddpm: DDPM,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    save_dir: str,
    image_shape: Tuple[int, int],
    n_sample: int = 5
) -> None:
    """
    Generate and save sample images at checkpoints.
    
    Args:
        ddpm: The diffusion model
        train_dataloader: Training data loader
        valid_dataloader: Validation data loader
        device: Device to run generation on
        epoch: Current epoch number
        save_dir: Directory to save samples
        image_shape: Shape of the images
        n_sample: Number of samples per conditioning vector
    """
    # Number of datapoints to visualize (at most 10)
    n_datapoint = min(10, train_dataloader.batch_size)
    
    def sample_save_plots(dataloader: DataLoader, dataset_type: str) -> None:
        """Helper function to generate and save plots for a dataset."""
        # Get a batch of data
        data_batch = next(iter(dataloader))
        
        # Extract data
        x_real = data_batch['y'][:n_datapoint].to(device)
        c_real = data_batch['x'][:n_datapoint].to(device)
        ffi_nums = data_batch['ffi_num'][:n_datapoint]
        orbits = data_batch['orbit'][:n_datapoint]
        
        # Generate samples
        x_gen, x_gen_store = ddpm.sample_c(
            c_real, 
            n_sample, 
            (1, image_shape[0], image_shape[1]), 
            device
        )
        
        # Create figure with original images and generated samples
        fig, axes = plt.subplots(n_datapoint, n_sample + 1, figsize=(15, 30))
        plt.subplots_adjust(top=1.7)
        
        # Combine originals and generated samples for display
        x_all = torch.cat([
            torch.cat([x_real[i:i+1], x_gen[i*n_sample:(i+1)*n_sample]])
            for i in range(n_datapoint)
        ])
        
        # Plot all images
        for idx, img in enumerate(x_all):
            row = idx // (n_sample + 1)
            col = idx % (n_sample + 1)
            
            # Display image
            axes[row, col].imshow(
                img[0].cpu().detach().numpy(), 
                cmap='gray', 
                vmin=0, 
                vmax=1
            )
            axes[row, col].axis('off')
            
            # Set column titles
            if row == 0:
                if col == 0:
                    axes[row, col].set_title("Original", fontsize=12)
                else:
                    axes[row, col].set_title(f"Sample {col}", fontsize=12)
            
            # Set row titles (FFI info)
            if col == 0:
                axes[row, col].set_title(
                    f"Orbit {orbits[row]}, FFI {ffi_nums[row]}", 
                    fontsize=10, 
                    loc='left'
                )
        
        # Set overall title
        fig.suptitle(f"{dataset_type} Predictions (Epoch {epoch})", fontsize=25)
        
        # Save figure
        plt.tight_layout()
        fig_path = os.path.join(save_dir, f"image_ep{epoch}_{dataset_type.lower()}.pdf")
        fig.savefig(fig_path)
        print(f'Saved {dataset_type} samples to {fig_path}')
        plt.close(fig)
    
    # Generate and save samples for both datasets
    sample_save_plots(train_dataloader, "Training")
    sample_save_plots(valid_dataloader, "Validation")


def save_loss_plots(
    loss_history_train: List[float],
    loss_history_valid: List[float],
    save_dir: str
) -> None:
    """
    Save plots of training and validation loss.
    
    Args:
        loss_history_train: List of training losses
        loss_history_valid: List of validation losses
        save_dir: Directory to save plots
    """
    # Plot full loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history_valid, label="Validation Loss")
    plt.plot(loss_history_train, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'loss_graph.png'))
    plt.close()
    
    # Plot last 50 epochs (if available)
    if len(loss_history_valid) >= 50:
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(len(loss_history_valid) - 50, len(loss_history_valid)), 
            loss_history_valid[-50:], 
            label="Validation Loss"
        )
        plt.plot(
            range(len(loss_history_valid) - 50, len(loss_history_valid)), 
            loss_history_train[-50:], 
            label="Training Loss"
        )
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training and Validation MSE Loss (Last 50 Epochs)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'loss_graph_last50.png'))
        plt.close()


def save_loss_history(
    loss_history_train: List[float],
    loss_history_valid: List[float],
    time_history: List[float],
    save_dir: str
) -> None:
    """
    Save loss history to a text file.
    
    Args:
        loss_history_train: List of training losses
        loss_history_valid: List of validation losses
        time_history: List of cumulative training times (hours)
        save_dir: Directory to save the history
    """
    with open(os.path.join(save_dir, 'loss_history.txt'), 'w') as f:
        f.write("Epoch\tTraining MSE Loss\tValidation MSE Loss\tTime(Hrs)\n")
        
        for i in range(len(loss_history_valid)):
            f.write(
                f"{i}\t\t{loss_history_train[i]:.4e}\t\t"
                f"{loss_history_valid[i]:.4e}\t\t{time_history[i]}\n"
            )
        
    print(f"Loss history saved to {os.path.join(save_dir, 'loss_history.txt')}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a diffusion model on TESS images")
    
    # Data paths
    parser.add_argument('--save_dir', type=str, default='model_TESS_diffusion/',
                        help='Directory to save model and results')
    parser.add_argument('--angle_filename', type=str, default="angles_O11-54_data_dic.pkl",
                        help='Filename for angle data')
    parser.add_argument('--ccd_folder', type=str, 
                        default="/pdo/users/jlupoiii/TESS/data/processed_images_im64x64/",
                        help='Folder containing processed images')
    
    # Training parameters
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training/validation ratio')
    parser.add_argument('--n_T', type=int, default=600,
                        help='Number of diffusion timesteps')
    parser.add_argument('--n_feat', type=int, default=256,
                        help='Number of features in U-Net')
    parser.add_argument('--lrate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=150,
                        help='Early stopping patience')
    
    # Checkpoint parameters
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoints')
    parser.add_argument('--epoch_checkpoint', type=int, default=100,
                        help='Save samples every N epochs')
    parser.add_argument('--checkpoint_gpu', type=int, default=0,
                        help='GPU ID to use for checkpoints')
    
    # Data processing
    parser.add_argument('--image_height', type=int, default=64,
                        help='Image height')
    parser.add_argument('--image_width', type=int, default=64,
                        help='Image width')
    parser.add_argument('--num_processes', type=int, default=80,
                        help='Number of processes for data loading')
    
    return parser.parse_args()


def main():
    """Main function to set up and launch distributed training."""
    # Parse arguments
    args = parse_args()
    
    # Set up configuration
    config = {
        'save_dir': args.save_dir,
        'angle_filename': args.angle_filename,
        'ccd_folder': args.ccd_folder,
        'image_shape': (args.image_height, args.image_width),
        'num_processes': args.num_processes,
        'n_epoch': args.n_epoch,
        'batch_size': args.batch_size,
        'train_ratio': args.train_ratio,
        'n_T': args.n_T,
        'n_feat': args.n_feat,
        'lrate': args.lrate,
        'save_model': args.save_model,
        'epoch_checkpoint': args.epoch_checkpoint,
        'checkpoint_gpu': args.checkpoint_gpu,
        'patience': args.patience,
    }
    
    # Save model configuration
    save_model_info(config['save_dir'], config)
    
    # Prepare dataset (this must be done in the main process)
    train_dataset, valid_dataset = prepare_dataset(
        angle_filename=config['angle_filename'],
        ccd_folder=config['ccd_folder'],
        image_shape=config['image_shape'],
        num_processes=config['num_processes'],
        train_ratio=config['train_ratio'],
        save_dir=config['save_dir']
    )
    
    # Add datasets to config
    config['train_dataset'] = train_dataset
    config['valid_dataset'] = valid_dataset
    
    # Get number of available GPUs
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1, f"Requires at least 1 GPU to run, but got {n_gpus}"
    
    # Launch distributed training
    world_size = n_gpus
    print(f'Using {world_size} GPUs for training')
    
    mp.spawn(
        train,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()