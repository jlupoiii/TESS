"""
Dataset classes for TESS image data.

This module contains dataset implementations for loading and processing
TESS (Transiting Exoplanet Survey Satellite) image data.
"""

import os
import time
import pickle
import multiprocessing
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from PIL import Image
from astropy.io import fits


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
            for x, y, ffi_num in results:
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


def create_train_valid_datasets(
    dataset: Dataset, 
    train_split_criteria: callable, 
    valid_split_criteria: callable
) -> Tuple[Subset, Subset]:
    """
    Create training and validation datasets from a full dataset.
    
    Args:
        dataset: The complete dataset
        train_split_criteria: Function to determine if a sample belongs to the training set
        valid_split_criteria: Function to determine if a sample belongs to the validation set
        
    Returns:
        Tuple of (training dataset, validation dataset)
    """
    train_indices = [
        idx for idx in range(len(dataset)) 
        if train_split_criteria(dataset[idx])
    ]
    
    valid_indices = [
        idx for idx in range(len(dataset)) 
        if valid_split_criteria(dataset[idx])
    ]
    
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    
    return train_dataset, valid_dataset


class TESS_4096_original_images(Dataset):
    """
    Dataset for raw original 4096x4096 TESS images.
    """
    def __init__(self):
        self.ffi_to_fits_filepath = {}
        
        # Build index of fits files
        fits_folder_paths = [f"/pdo/users/roland/SL_data/O{i}_data/" for i in range(9, 63)]

        for fits_folder_path in fits_folder_paths:
            orbit = fits_folder_path[27:29] if fits_folder_path[29] == '_' else fits_folder_path[27:28]
            for fits_filename in os.listdir(fits_folder_path):
                # Only include camera 3 FITS files
                if (len(fits_filename) > 40 and 
                    fits_filename[-7:] == 'fits.gz' and 
                    fits_filename[27] == '3'):
                    ffi_num = fits_filename[18:26]
                    self.ffi_to_fits_filepath[ffi_num] = os.path.join(fits_folder_path, fits_filename)

    def __len__(self) -> int:
        return len(self.ffi_to_fits_filepath)

    def __getitem__(self, ffi_num: str) -> torch.Tensor:
        # Rows and columns to remove from the image (black areas)
        rows_to_delete = range(2048, 2108)
        columns_to_delete = list(range(0, 44)) + list(range(2092, 2180)) + list(range(4228, 4272))
        
        # Load and process image
        image = fits.getdata(self.ffi_to_fits_filepath[ffi_num], ext=0)
        image = np.delete(image, rows_to_delete, axis=0)
        image = np.delete(image, columns_to_delete, axis=1)
        image = image.astype(np.float64)
        image *= 1/633118  # Scale values to reasonable range
        
        return torch.tensor(image)


class TESS_4096_processed_images(Dataset):
    """
    Dataset for pre-processed 4096x4096 TESS images.
    """
    def __init__(self):
        self.ffi_to_pkl_filepath = {}
        
        pkl_folder_path = "/pdo/users/jlupoiii/TESS/data/processed_images_im4096x4096/"
        for pkl_filename in os.listdir(pkl_folder_path):
            ffi_num = pkl_filename[18:26]
            self.ffi_to_pkl_filepath[ffi_num] = os.path.join(pkl_folder_path, pkl_filename)

    def __len__(self) -> int:
        return len(self.ffi_to_pkl_filepath)

    def __getitem__(self, ffi_num: str) -> torch.Tensor:
        with open(self.ffi_to_pkl_filepath[ffi_num], 'rb') as file:
            return pickle.load(file)