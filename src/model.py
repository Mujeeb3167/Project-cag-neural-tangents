# =============================================================
# Neural Network Architecture Module
# AI-Based Neural Network System for Function Approximation
# Author: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)
# =============================================================

"""
Professional feedforward neural network architectures for
function approximation, circle learning, and PINN.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)


# =============================================================
# Device Detection
# =============================================================

def get_device() -> torch.device:
    """
    Detect and return the best available compute device.
    
    Returns:
        torch.device: CUDA device if GPU is available, else CPU.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU (no GPU detected)")
    return device


# =============================================================
# Primary Function Approximator
# =============================================================

class FunctionApproximator(nn.Module):
    """
    Feedforward Neural Network for function approximation.
    
    Architecture:
        Input(1) → Dense(128, ReLU) → Dense(128, ReLU) →
        Dense(64, ReLU) → Dense(32, ReLU) → Output(1)
    
    Features:
        - Xavier/Glorot weight initialization
        - Configurable hidden layer sizes
        - ReLU or Tanh activation functions
        - Batch normalization support (optional)
    
    Attributes:
        network: The sequential neural network.
        input_dim: Dimension of input features.
        output_dim: Dimension of output.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
        output_dim: int = 1,
        activation: str = 'relu',
        use_batch_norm: bool = False
    ) -> None:
        """
        Initialize the function approximator network.
        
        Args:
            input_dim: Number of input features.
            hidden_layers: List of hidden layer sizes. Default: [128, 128, 64, 32].
            output_dim: Number of output features.
            activation: Activation function ('relu' or 'tanh').
            use_batch_norm: Whether to include batch normalization layers.
        """
        super(FunctionApproximator, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128, 64, 32]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        # Select activation function
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU
        }
        if activation.lower() not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}. Use: {list(activation_map.keys())}")
        act_fn = activation_map[activation.lower()]
        
        # Build the network layers
        layers = []
        prev_size = input_dim
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_fn())
            prev_size = hidden_size
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Apply Xavier initialization
        self.apply(self._init_weights)
        
        # Log architecture
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"FunctionApproximator created: {hidden_layers}, "
            f"activation={activation}, params={total_params:,} "
            f"(trainable={trainable_params:,})"
        )
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Apply Xavier/Glorot uniform initialization to Linear layers.
        
        Args:
            module: The neural network module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)
    
    def get_architecture_summary(self) -> str:
        """Return a human-readable summary of the architecture."""
        summary_lines = ["FunctionApproximator Architecture:"]
        summary_lines.append(f"  Input:  {self.input_dim}")
        for i, h in enumerate(self.hidden_layers):
            summary_lines.append(f"  Hidden {i+1}: {h}")
        summary_lines.append(f"  Output: {self.output_dim}")
        total = sum(p.numel() for p in self.parameters())
        summary_lines.append(f"  Total Parameters: {total:,}")
        return "\n".join(summary_lines)


# =============================================================
# Circle Approximator (Implicit Function)
# =============================================================

class CircleApproximator(nn.Module):
    """
    Neural network for learning the implicit circle equation x² + y² = r².
    
    Maps x → y for the upper semicircle (y = sqrt(r² - x²)).
    Uses the same architecture pattern as FunctionApproximator.
    
    Attributes:
        network: The sequential neural network.
        radius: The radius of the target circle.
    """
    
    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        activation: str = 'relu',
        radius: float = 5.0
    ) -> None:
        """
        Initialize the circle approximator.
        
        Args:
            hidden_layers: List of hidden layer sizes.
            activation: Activation function name.
            radius: Radius of the target circle.
        """
        super(CircleApproximator, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128, 64, 32]
        
        self.radius = radius
        self.hidden_layers = hidden_layers
        
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
        }
        act_fn = activation_map.get(activation.lower(), nn.ReLU)
        
        layers = []
        prev_size = 1  # x input
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_fn())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))  # y output
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"CircleApproximator created: {hidden_layers}, radius={radius}, params={total_params:,}")
    
    def _init_weights(self, module: nn.Module) -> None:
        """Apply Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x → y (predicting y-coordinate on circle)."""
        return self.network(x)


# =============================================================
# Physics-Informed Neural Network (PINN)
# =============================================================

class PINNApproximator(nn.Module):
    """
    Physics-Informed Neural Network for function approximation.
    
    Same architecture as FunctionApproximator, but designed to be
    trained with a physics-informed loss that includes the derivative
    constraint: the network output's derivative should match the
    known analytic derivative.
    
    Attributes:
        network: The sequential neural network.
    """
    
    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        activation: str = 'tanh'
    ) -> None:
        """
        Initialize the PINN approximator.
        
        Args:
            hidden_layers: List of hidden layer sizes.
            activation: Activation function ('tanh' recommended for PINNs).
        """
        super(PINNApproximator, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128, 64, 32]
        
        self.hidden_layers = hidden_layers
        
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
        }
        act_fn = activation_map.get(activation.lower(), nn.Tanh)
        
        layers = []
        prev_size = 1
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_fn())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"PINNApproximator created: {hidden_layers}, activation={activation}, params={total_params:,}")
    
    def _init_weights(self, module: nn.Module) -> None:
        """Apply Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PINN."""
        return self.network(x)


# =============================================================
# Model Persistence
# =============================================================

def save_model(
    model: nn.Module,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save model checkpoint including state dict and optional metadata.
    
    Args:
        model: The PyTorch model to save.
        filepath: Path to save the checkpoint.
        metadata: Optional metadata (epoch, loss, architecture info).
    
    Returns:
        The full path to the saved checkpoint.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    # Store architecture info if available
    if hasattr(model, 'hidden_layers'):
        checkpoint['hidden_layers'] = model.hidden_layers
    if hasattr(model, 'input_dim'):
        checkpoint['input_dim'] = model.input_dim
    if hasattr(model, 'output_dim'):
        checkpoint['output_dim'] = model.output_dim
    
    torch.save(checkpoint, filepath)
    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(
    model_class: type,
    filepath: str,
    device: Optional[torch.device] = None,
    **kwargs
) -> nn.Module:
    """
    Load a model from a checkpoint file.
    
    Args:
        model_class: The class of the model to instantiate.
        filepath: Path to the checkpoint file.
        device: Device to load the model onto.
        **kwargs: Additional keyword arguments for model construction.
    
    Returns:
        The loaded model in evaluation mode.
    
    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Reconstruct model with saved architecture if available
    if 'hidden_layers' in checkpoint and 'hidden_layers' not in kwargs:
        kwargs['hidden_layers'] = checkpoint['hidden_layers']
    
    model = model_class(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {filepath} (class={checkpoint.get('model_class', 'unknown')})")
    return model
