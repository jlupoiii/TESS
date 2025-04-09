"""
Configuration utilities for the BEAM project.

This module contains functions for loading and managing configurations.
"""

import os
import yaml
from typing import Dict, Any, Optional
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def flatten_config(config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Flatten a nested configuration dictionary.
    
    Args:
        config: Nested configuration dictionary
        prefix: Prefix for flattened keys
        
    Returns:
        Flattened configuration dictionary
    """
    flat_config = {}
    
    for key, value in config.items():
        new_key = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            flat_nested = flatten_config(value, f"{new_key}_")
            flat_config.update(flat_nested)
        else:
            flat_config[new_key] = value
    
    return flat_config


def prepare_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare configuration for training, handling auto settings.
    
    Args:
        config: Raw configuration
        
    Returns:
        Processed configuration for training
    """
    # Create a flattened copy
    flat_config = flatten_config(config)
    
    # Auto-detect GPU count if not specified
    if not flat_config['distributed_world_size']:
        flat_config['distributed_world_size'] = torch.cuda.device_count()
    
    # Ensure save directory exists
    os.makedirs(flat_config['paths_save_dir'], exist_ok=True)
    
    # Save a copy of the config
    save_config(config, os.path.join(flat_config['paths_save_dir'], 'config.yaml'))
    
    return flat_config
