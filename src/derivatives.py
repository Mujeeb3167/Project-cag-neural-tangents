# =============================================================
# Derivative Computation Module
# AI-Based Neural Network System for Function Approximation
# Author: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)
# =============================================================

"""
Derivative computation using automatic differentiation (PyTorch autograd)
and finite difference approximation (central difference method).
"""

import torch
import numpy as np
from typing import Callable, Tuple, Optional, Union, Dict
import logging

logger = logging.getLogger(__name__)


# =============================================================
# Automatic Differentiation Calculator
# =============================================================

class AutoDiffCalculator:
    """
    Computes derivatives using PyTorch automatic differentiation (autograd).
    
    This is the PRIMARY derivative computation method. It computes
    exact analytical derivatives of the neural network function by
    backpropagating through the computational graph.
    
    Attributes:
        model: The neural network model.
        device: The compute device (CPU/GPU).
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize the automatic differentiation calculator.
        
        Args:
            model: Trained PyTorch model for function approximation.
            device: Device for computation. Defaults to CPU.
        """
        self.model = model
        self.device = device if device is not None else torch.device('cpu')
        self.model.eval()
        logger.info("AutoDiffCalculator initialized.")
    
    def compute_derivative(
        self,
        x: Union[float, np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the derivative of the neural network at given point(s).
        
        Uses torch.autograd.grad to compute dy/dx for the model.
        
        Args:
            x: Input value(s) — scalar, numpy array, or torch tensor.
        
        Returns:
            Tuple of (function_values, derivative_values) as numpy arrays.
        """
        # Convert to tensor
        if isinstance(x, (int, float)):
            x_tensor = torch.FloatTensor([[x]]).to(self.device)
        elif isinstance(x, np.ndarray):
            x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(self.device)
        else:
            x_tensor = x.clone().reshape(-1, 1).to(self.device)
        
        x_tensor.requires_grad_(True)
        
        # Forward pass
        y_pred = self.model(x_tensor)
        
        # Compute gradients using autograd
        dy_dx = torch.autograd.grad(
            outputs=y_pred,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Convert to numpy
        f_values = y_pred.detach().cpu().numpy().flatten()
        d_values = dy_dx.detach().cpu().numpy().flatten()
        
        logger.debug(f"AutoDiff computed for {len(f_values)} points.")
        return f_values, d_values
    
    def compute_derivative_at_point(
        self,
        x0: float
    ) -> Tuple[float, float]:
        """
        Compute f(x0) and f'(x0) at a single point using autograd.
        
        Args:
            x0: The point at which to compute the derivative.
        
        Returns:
            Tuple of (f_x0, f_prime_x0) as float values.
        """
        f_vals, d_vals = self.compute_derivative(x0)
        f_x0 = float(f_vals[0])
        f_prime_x0 = float(d_vals[0])
        logger.info(f"AutoDiff at x0={x0:.4f}: f(x0)={f_x0:.6f}, f'(x0)={f_prime_x0:.6f}")
        return f_x0, f_prime_x0


# =============================================================
# Finite Difference Calculator
# =============================================================

class FiniteDifferenceCalculator:
    """
    Computes derivatives using the central finite difference approximation.
    
    Central difference formula:
        f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
    
    This is the SECONDARY comparison method.
    
    Attributes:
        func: The function to differentiate (callable).
        h: Step size for finite differences.
    """
    
    def __init__(
        self,
        func: Callable,
        h: float = 1e-5
    ) -> None:
        """
        Initialize the finite difference calculator.
        
        Args:
            func: Function to compute derivatives for.
            h: Step size for the central difference (default: 1e-5).
        """
        self.func = func
        self.h = h
        logger.info(f"FiniteDifferenceCalculator initialized with h={h:.1e}")
    
    def compute_derivative(
        self,
        x: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Compute the central difference approximation of f'(x).
        
        f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
        
        Args:
            x: Input value(s) at which to compute the derivative.
        
        Returns:
            Array of derivative approximations.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        f_plus = self.func(x + self.h)
        f_minus = self.func(x - self.h)
        derivative = (f_plus - f_minus) / (2 * self.h)
        logger.debug(f"Finite difference computed for {len(x)} points.")
        return derivative.astype(np.float32)
    
    def compute_derivative_at_point(
        self,
        x0: float
    ) -> Tuple[float, float]:
        """
        Compute f(x0) and f'(x0) at a single point.
        
        Args:
            x0: The point at which to compute the derivative.
        
        Returns:
            Tuple of (f_x0, f_prime_x0) as float values.
        """
        f_x0 = float(self.func(np.array([x0]))[0])
        f_prime_x0 = float(self.compute_derivative(np.array([x0]))[0])
        logger.info(f"FiniteDiff at x0={x0:.4f}: f(x0)={f_x0:.6f}, f'(x0)={f_prime_x0:.6f}")
        return f_x0, f_prime_x0


# =============================================================
# Neural Network Finite Difference
# =============================================================

class NNFiniteDifferenceCalculator:
    """
    Computes derivatives of the neural network using finite differences.
    
    Instead of using autograd, this evaluates the NN at x+h and x-h
    and uses the central difference formula on the NN outputs.
    
    Useful for comparing autograd results with finite differences
    applied to the same neural network.
    
    Attributes:
        model: The neural network model.
        device: Compute device.
        h: Step size.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        h: float = 1e-5
    ) -> None:
        """
        Initialize the NN finite difference calculator.
        
        Args:
            model: Trained neural network.
            device: Compute device.
            h: Step size for central differences.
        """
        self.model = model
        self.device = device if device is not None else torch.device('cpu')
        self.h = h
        self.model.eval()
    
    def compute_derivative(
        self,
        x: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute NN derivative using central differences.
        
        Args:
            x: Input values.
        
        Returns:
            Tuple of (function_values, derivative_values).
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float32))
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(self.device)
            x_plus = torch.FloatTensor((x + self.h).reshape(-1, 1)).to(self.device)
            x_minus = torch.FloatTensor((x - self.h).reshape(-1, 1)).to(self.device)
            
            f_x = self.model(x_tensor).cpu().numpy().flatten()
            f_plus = self.model(x_plus).cpu().numpy().flatten()
            f_minus = self.model(x_minus).cpu().numpy().flatten()
        
        derivative = (f_plus - f_minus) / (2 * self.h)
        return f_x, derivative


# =============================================================
# Derivative Comparison
# =============================================================

def compare_derivatives(
    x: np.ndarray,
    true_derivative: np.ndarray,
    auto_diff_derivative: np.ndarray,
    finite_diff_derivative: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compare derivatives from different computation methods.
    
    Computes MSE, MAE, and max absolute error for:
    - Autograd vs true derivative
    - Finite difference vs true derivative
    - Autograd vs finite difference
    
    Args:
        x: Input x-values (for reference).
        true_derivative: True analytic derivative values.
        auto_diff_derivative: Autograd-computed derivative values.
        finite_diff_derivative: Finite difference derivative values.
    
    Returns:
        Dictionary with comparison metrics for each method pair.
    """
    def _compute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute error metrics between two arrays."""
        diff = np.abs(y_true - y_pred)
        return {
            'mse': float(np.mean((y_true - y_pred) ** 2)),
            'mae': float(np.mean(diff)),
            'max_error': float(np.max(diff)),
            'rmse': float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        }
    
    results = {
        'autograd_vs_true': _compute_errors(true_derivative, auto_diff_derivative),
        'finite_diff_vs_true': _compute_errors(true_derivative, finite_diff_derivative),
        'autograd_vs_finite_diff': _compute_errors(auto_diff_derivative, finite_diff_derivative)
    }
    
    logger.info("Derivative comparison completed:")
    for method, metrics in results.items():
        logger.info(f"  {method}: MSE={metrics['mse']:.2e}, MAE={metrics['mae']:.2e}")
    
    return results


def compute_all_derivatives(
    model: torch.nn.Module,
    x: np.ndarray,
    true_func: Callable,
    true_deriv_func: Callable,
    device: Optional[torch.device] = None
) -> Dict[str, np.ndarray]:
    """
    Compute derivatives using all available methods for comparison.
    
    Args:
        model: Trained neural network.
        x: Input x-values.
        true_func: True analytic function.
        true_deriv_func: True analytic derivative function.
        device: Compute device.
    
    Returns:
        Dictionary with derivative arrays from each method.
    """
    device = device if device is not None else torch.device('cpu')
    
    # True analytic derivative
    true_deriv = true_deriv_func(x)
    
    # Automatic differentiation
    auto_diff = AutoDiffCalculator(model, device)
    _, ad_deriv = auto_diff.compute_derivative(x)
    
    # Finite difference on true function
    fd_calc = FiniteDifferenceCalculator(true_func)
    fd_deriv = fd_calc.compute_derivative(x)
    
    # Finite difference on neural network
    nn_fd_calc = NNFiniteDifferenceCalculator(model, device)
    _, nn_fd_deriv = nn_fd_calc.compute_derivative(x)
    
    results = {
        'x': x,
        'true_derivative': true_deriv,
        'autograd_derivative': ad_deriv,
        'finite_diff_derivative': fd_deriv,
        'nn_finite_diff_derivative': nn_fd_deriv
    }
    
    logger.info(f"All derivatives computed for {len(x)} points using 4 methods.")
    return results
