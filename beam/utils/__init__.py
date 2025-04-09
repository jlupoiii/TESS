"""
Utility functions for BEAM.

This module provides visualization, configuration, and I/O utilities.
"""

from beam.utils.visualization import (
    plot_samples,
    plot_loss_history,
    plot_generation_process
)

from beam.utils.config import (
    load_config,
    save_config,
    flatten_config,
    prepare_training_config
)

__all__ = [
    'plot_samples',
    'plot_loss_history',
    'plot_generation_process',
    'load_config',
    'save_config',
    'flatten_config',
    'prepare_training_config'
]