# =============================================================
# Dataset Generation Module
# AI-Based Neural Network System for Function Approximation
# Author: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)
# =============================================================

"""
Synthetic dataset generation for function approximation.
Supports noise-free and noisy data generation with train/val splits.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, Callable, Dict
import os
import logging

logger = logging.getLogger(__name__)


# =============================================================
# Reproducibility
# =============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across all libraries for full reproducibility.
    
    Args:
        seed: Integer seed value for random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility.")


# =============================================================
# Target Functions
# =============================================================

def explicit_function(x: np.ndarray) -> np.ndarray:
    """
    Primary explicit function: f(x) = x³ - 2x² + sin(x).
    
    Args:
        x: Input array of x-values.
    
    Returns:
        Array of f(x) values.
    """
    return x ** 3 - 2 * x ** 2 + np.sin(x)


def explicit_derivative(x: np.ndarray) -> np.ndarray:
    """
    Analytic derivative: f'(x) = 3x² - 4x + cos(x).
    
    Args:
        x: Input array of x-values.
    
    Returns:
        Array of f'(x) values.
    """
    return 3 * x ** 2 - 4 * x + np.cos(x)


def explicit_function_torch(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-compatible version of f(x) = x³ - 2x² + sin(x).
    Used for computing true function values on tensors.
    
    Args:
        x: Input tensor.
    
    Returns:
        Tensor of f(x) values.
    """
    return x ** 3 - 2 * x ** 2 + torch.sin(x)


def explicit_derivative_torch(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-compatible version of f'(x) = 3x² - 4x + cos(x).
    
    Args:
        x: Input tensor.
    
    Returns:
        Tensor of f'(x) values.
    """
    return 3 * x ** 2 - 4 * x + torch.cos(x)


# =============================================================
# PyTorch Dataset
# =============================================================

class FunctionDataset(Dataset):
    """
    PyTorch Dataset wrapper for function approximation data.
    
    Stores (x, y) pairs as float32 tensors with shape (N, 1).
    
    Attributes:
        x_tensor: Input values as a 2D tensor of shape (N, 1).
        y_tensor: Target values as a 2D tensor of shape (N, 1).
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the dataset.
        
        Args:
            x: Input values as a 1D numpy array.
            y: Target values as a 1D numpy array.
        
        Raises:
            ValueError: If x and y have different lengths.
        """
        if len(x) != len(y):
            raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")
        
        self.x_tensor = torch.FloatTensor(x.reshape(-1, 1))
        self.y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        logger.debug(f"FunctionDataset created with {len(x)} samples.")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.x_tensor)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the (x, y) pair at the given index."""
        return self.x_tensor[idx], self.y_tensor[idx]


# =============================================================
# Dataset Generation
# =============================================================

def generate_dataset(
    func: Callable = None,
    x_min: float = -3.0,
    x_max: float = 3.0,
    n_samples: int = 1000,
    noise_std: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset from an analytic function.
    
    Creates evenly spaced x-values in [x_min, x_max] and computes
    y = func(x) + optional Gaussian noise.
    
    Args:
        func: The target function. Defaults to explicit_function.
        x_min: Minimum x-value for the domain.
        x_max: Maximum x-value for the domain.
        n_samples: Number of data points to generate.
        noise_std: Standard deviation of Gaussian noise (0 = no noise).
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (x, y) numpy arrays.
    """
    set_seed(seed)
    
    if func is None:
        func = explicit_function
    
    # Generate evenly spaced x-values
    x = np.linspace(x_min, x_max, n_samples).astype(np.float32)
    
    # Compute clean function values
    y = func(x).astype(np.float32)
    
    # Add Gaussian noise if specified
    if noise_std > 0.0:
        noise = np.random.normal(0.0, noise_std, size=y.shape).astype(np.float32)
        y_noisy = y + noise
        logger.info(
            f"Generated noisy dataset: {n_samples} samples on [{x_min}, {x_max}], "
            f"noise_std={noise_std:.4f}"
        )
        return x, y_noisy
    
    logger.info(
        f"Generated clean dataset: {n_samples} samples on [{x_min}, {x_max}]"
    )
    return x, y


def create_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders with train/validation split.
    
    Args:
        x: Input values as a numpy array.
        y: Target values as a numpy array.
        batch_size: Number of samples per batch.
        val_split: Fraction of data to use for validation (0-1).
        seed: Random seed for reproducibility of the split.
    
    Returns:
        Tuple of (train_loader, val_loader).
    
    Raises:
        ValueError: If val_split is not in (0, 1).
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")
    
    set_seed(seed)
    
    # Create the full dataset
    full_dataset = FunctionDataset(x, y)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Perform the split
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    logger.info(
        f"Created DataLoaders: train={train_size} samples, "
        f"val={val_size} samples, batch_size={batch_size}"
    )
    
    return train_loader, val_loader


# =============================================================
# Dataset Persistence
# =============================================================

def save_dataset(
    x: np.ndarray,
    y: np.ndarray,
    filepath: str,
    metadata: Optional[Dict] = None
) -> str:
    """
    Save dataset to a compressed .npz file.
    
    Args:
        x: Input values.
        y: Target values.
        filepath: Path to save the file (without extension).
        metadata: Optional metadata dictionary to include.
    
    Returns:
        The full path to the saved file.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    save_dict = {'x': x, 'y': y}
    if metadata:
        for key, value in metadata.items():
            save_dict[key] = np.array(value)
    
    full_path = filepath if filepath.endswith('.npz') else filepath + '.npz'
    np.savez_compressed(full_path, **save_dict)
    logger.info(f"Dataset saved to {full_path} ({len(x)} samples)")
    return full_path


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from a .npz file.
    
    Args:
        filepath: Path to the .npz file.
    
    Returns:
        Tuple of (x, y) numpy arrays.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    full_path = filepath if filepath.endswith('.npz') else filepath + '.npz'
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Dataset file not found: {full_path}")
    
    data = np.load(full_path)
    x, y = data['x'], data['y']
    logger.info(f"Dataset loaded from {full_path} ({len(x)} samples)")
    return x, y


# =============================================================
# Convenience Functions
# =============================================================

def generate_both_datasets(
    x_min: float = -3.0,
    x_max: float = 3.0,
    n_samples: int = 1000,
    noise_std: float = 0.05,
    seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate both clean and noisy datasets for comparison.
    
    Args:
        x_min: Minimum x-value.
        x_max: Maximum x-value.
        n_samples: Number of samples.
        noise_std: Standard deviation for noisy dataset.
        seed: Random seed.
    
    Returns:
        Dictionary with 'clean' and 'noisy' keys, each mapping to (x, y) tuples.
    """
    x_clean, y_clean = generate_dataset(
        func=explicit_function,
        x_min=x_min, x_max=x_max,
        n_samples=n_samples,
        noise_std=0.0,
        seed=seed
    )
    
    x_noisy, y_noisy = generate_dataset(
        func=explicit_function,
        x_min=x_min, x_max=x_max,
        n_samples=n_samples,
        noise_std=noise_std,
        seed=seed
    )
    
    logger.info(
        f"Generated both clean and noisy datasets: "
        f"{n_samples} samples each, noise_std={noise_std}"
    )
    
    return {
        'clean': (x_clean, y_clean),
        'noisy': (x_noisy, y_noisy)
    }
