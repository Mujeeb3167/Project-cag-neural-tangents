# =============================================================
# Implicit Curve / Conic Section Module
# AI-Based Neural Network System for Function Approximation
# Author: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)
# =============================================================

"""
Implicit curve handling for conic sections, specifically the circle
x² + y² = 25 (radius 5, center at origin).

Supports data generation, neural network training, and
implicit tangent line computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
import time
import copy
import logging
import os

logger = logging.getLogger(__name__)


# =============================================================
# Circle Curve Class
# =============================================================

class CircleCurve:
    """
    Represents the implicit circle x² + y² = r².
    
    Handles:
    - Parametric and explicit representations
    - Data generation for neural network training
    - Implicit differentiation (dy/dx = -x/y)
    - Tangent line computation
    
    Attributes:
        radius: Radius of the circle.
        center: Center coordinates (cx, cy).
    """
    
    def __init__(
        self,
        radius: float = 5.0,
        center: Tuple[float, float] = (0.0, 0.0)
    ) -> None:
        """
        Initialize the circle curve.
        
        Args:
            radius: Circle radius (default: 5.0).
            center: Center point (default: origin).
        """
        self.radius = radius
        self.center = center
        logger.info(f"CircleCurve initialized: r={radius}, center={center}")
    
    def equation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate the implicit equation: (x-cx)² + (y-cy)² - r² = 0.
        
        Returns 0 for points on the circle, positive outside, negative inside.
        
        Args:
            x: X-coordinates.
            y: Y-coordinates.
        
        Returns:
            Residual values (should be ~0 on the circle).
        """
        cx, cy = self.center
        return (x - cx) ** 2 + (y - cy) ** 2 - self.radius ** 2
    
    def y_from_x(self, x: np.ndarray, upper: bool = True) -> np.ndarray:
        """
        Compute y from x for the explicit parametrization.
        
        y = cy ± sqrt(r² - (x - cx)²)
        
        Args:
            x: X-coordinates.
            upper: If True, return upper semicircle (positive sqrt).
        
        Returns:
            Y-coordinates on the circle.
        
        Raises:
            ValueError: If any x is outside the circle.
        """
        cx, cy = self.center
        val = self.radius ** 2 - (x - cx) ** 2
        
        if np.any(val < -1e-10):
            invalid = x[val < -1e-10]
            logger.warning(f"Points outside circle domain: {invalid[:5]}")
        
        val = np.maximum(val, 0.0)  # Clamp for numerical stability
        sqrt_val = np.sqrt(val)
        
        if upper:
            return cy + sqrt_val
        else:
            return cy - sqrt_val
    
    def implicit_derivative(self, x0: float, y0: float) -> float:
        """
        Compute dy/dx at (x0, y0) using implicit differentiation.
        
        For x² + y² = r²:
            2x + 2y(dy/dx) = 0
            dy/dx = -x/y
        
        Args:
            x0: X-coordinate on the circle.
            y0: Y-coordinate on the circle.
        
        Returns:
            The derivative dy/dx at (x0, y0).
        
        Raises:
            ValueError: If y0 is zero (vertical tangent).
        """
        if abs(y0) < 1e-10:
            raise ValueError(f"Vertical tangent at y0={y0}: dy/dx is undefined.")
        
        cx, cy = self.center
        slope = -(x0 - cx) / (y0 - cy)
        logger.debug(f"Implicit derivative at ({x0:.4f}, {y0:.4f}): dy/dx = {slope:.6f}")
        return slope
    
    def tangent_at_point(
        self,
        x0: float,
        y0: Optional[float] = None,
        x_range: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute the tangent line at a point on the circle.
        
        Args:
            x0: X-coordinate of the tangent point.
            y0: Y-coordinate (computed from upper semicircle if None).
            x_range: X-values for evaluating the tangent line.
        
        Returns:
            Tangent line dictionary with slope, intercept, equation, etc.
        """
        if y0 is None:
            y0 = float(self.y_from_x(np.array([x0]), upper=True)[0])
        
        slope = self.implicit_derivative(x0, y0)
        intercept = y0 - slope * x0
        
        sign = "+" if intercept >= 0 else "-"
        equation = f"y = {slope:.4f}x {sign} {abs(intercept):.4f}"
        equation_point_form = f"y = {slope:.4f}(x - {x0:.4f}) + {y0:.4f}"
        
        result = {
            'x0': x0,
            'y0': y0,
            'f_x0': y0,
            'slope': slope,
            'intercept': intercept,
            'equation': equation,
            'equation_point_form': equation_point_form,
            'radius': self.radius,
            'method': 'implicit_differentiation',
            'x_range': x_range,
            'y_tangent': slope * (x_range - x0) + y0 if x_range is not None else None
        }
        
        logger.info(f"Circle tangent at ({x0:.4f}, {y0:.4f}): {equation_point_form}")
        return result
    
    def generate_dataset(
        self,
        n_samples: int = 500,
        noise_std: float = 0.0,
        upper_only: bool = True,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for the circle.
        
        Creates (x, y) pairs sampled from the circle. For the upper
        semicircle, x ∈ [-r, r] and y = sqrt(r² - x²).
        
        Args:
            n_samples: Number of data points.
            noise_std: Gaussian noise standard deviation.
            upper_only: If True, only upper semicircle.
            seed: Random seed.
        
        Returns:
            Tuple of (x, y) numpy arrays.
        """
        np.random.seed(seed)
        cx, cy = self.center
        
        if upper_only:
            # Slightly inside the domain to avoid sqrt(0)
            x = np.linspace(
                cx - self.radius + 0.01,
                cx + self.radius - 0.01,
                n_samples
            ).astype(np.float32)
            y = self.y_from_x(x, upper=True).astype(np.float32)
        else:
            # Full circle using parametric form
            theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False).astype(np.float32)
            x = (cx + self.radius * np.cos(theta)).astype(np.float32)
            y = (cy + self.radius * np.sin(theta)).astype(np.float32)
        
        # Add noise
        if noise_std > 0.0:
            y = y + np.random.normal(0, noise_std, size=y.shape).astype(np.float32)
            logger.info(f"Circle dataset: {n_samples} samples, noise_std={noise_std}")
        else:
            logger.info(f"Circle dataset: {n_samples} samples (noise-free)")
        
        return x, y
    
    def generate_full_circle_points(
        self,
        n_points: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate points for full circle visualization.
        
        Uses parametric form: x = r*cos(θ), y = r*sin(θ).
        
        Args:
            n_points: Number of points along the circle.
        
        Returns:
            Tuple of (x, y) arrays for plotting.
        """
        cx, cy = self.center
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = cx + self.radius * np.cos(theta)
        y = cy + self.radius * np.sin(theta)
        return x.astype(np.float32), y.astype(np.float32)


# =============================================================
# Circle Neural Network Trainer
# =============================================================

class CircleTrainer:
    """
    Trains a neural network to learn the circle mapping x → y.
    
    Uses the upper semicircle (y ≥ 0) for training, with
    Adam optimizer and early stopping.
    
    Attributes:
        model: Neural network model.
        device: Compute device.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize the circle trainer.
        
        Args:
            model: Neural network for circle approximation.
            device: Compute device (auto-detects if None).
        """
        self.device = device if device is not None else torch.device('cpu')
        self.model = model.to(self.device)
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}
        logger.info(f"CircleTrainer initialized on {self.device}")
    
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 32,
        val_split: float = 0.2,
        patience: int = 50,
        verbose: bool = True,
        log_interval: int = 100
    ) -> Dict[str, List[float]]:
        """
        Train the circle approximation model.
        
        Args:
            x_train: Input x-values.
            y_train: Target y-values (upper semicircle).
            epochs: Maximum training epochs.
            lr: Learning rate.
            batch_size: Mini-batch size.
            val_split: Validation split fraction.
            patience: Early stopping patience.
            verbose: Print progress.
            log_interval: Logging interval.
        
        Returns:
            Training history dictionary.
        """
        # Prepare data
        n_total = len(x_train)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        
        indices = np.random.permutation(n_total)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        x_tr = torch.FloatTensor(x_train[train_idx].reshape(-1, 1)).to(self.device)
        y_tr = torch.FloatTensor(y_train[train_idx].reshape(-1, 1)).to(self.device)
        x_va = torch.FloatTensor(x_train[val_idx].reshape(-1, 1)).to(self.device)
        y_va = torch.FloatTensor(y_train[val_idx].reshape(-1, 1)).to(self.device)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7
        )
        criterion = nn.MSELoss()
        
        # Early stopping
        best_val_loss = float('inf')
        best_model_state = None
        es_counter = 0
        best_epoch = 0
        
        start_time = time.time()
        logger.info(f"Training circle model: {epochs} epochs, lr={lr}")
        
        for epoch in range(1, epochs + 1):
            # Training
            self.model.train()
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                x_batch = x_tr[idx]
                y_batch = y_tr[idx]
                
                pred = self.model(x_batch)
                loss = criterion(pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / max(n_batches, 1)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(x_va)
                val_loss = criterion(val_pred, y_va).item()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and (epoch % log_interval == 0 or epoch == 1):
                elapsed = time.time() - start_time
                logger.info(
                    f"Circle Epoch {epoch:4d}/{epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best circle model from epoch {best_epoch}")
        
        total_time = time.time() - start_time
        logger.info(f"Circle training complete in {total_time:.1f}s. Best val: {best_val_loss:.6f}")
        
        return self.history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict y-values for given x-values.
        
        Args:
            x: Input x-values.
        
        Returns:
            Predicted y-values as numpy array.
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(self.device)
            y_pred = self.model(x_tensor).cpu().numpy().flatten()
        return y_pred
    
    def compute_ai_tangent(
        self,
        x0: float,
        x_range: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute the AI tangent at x0 using the trained model + autograd.
        
        Args:
            x0: X-coordinate of the tangent point.
            x_range: X-values for tangent line evaluation.
        
        Returns:
            Tangent line dictionary.
        """
        self.model.eval()
        
        x_tensor = torch.FloatTensor([[x0]]).to(self.device)
        x_tensor.requires_grad_(True)
        
        y_pred = self.model(x_tensor)
        dy_dx = torch.autograd.grad(
            outputs=y_pred,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=False
        )[0]
        
        y0 = float(y_pred.item())
        slope = float(dy_dx.item())
        intercept = y0 - slope * x0
        
        sign = "+" if intercept >= 0 else "-"
        equation = f"y = {slope:.4f}x {sign} {abs(intercept):.4f}"
        equation_point_form = f"y = {slope:.4f}(x - {x0:.4f}) + {y0:.4f}"
        
        result = {
            'x0': x0,
            'y0': y0,
            'f_x0': y0,
            'slope': slope,
            'intercept': intercept,
            'equation': equation,
            'equation_point_form': equation_point_form,
            'method': 'ai_circle_autograd',
            'x_range': x_range,
            'y_tangent': slope * (x_range - x0) + y0 if x_range is not None else None
        }
        
        logger.info(f"AI circle tangent at x0={x0}: y0={y0:.4f}, slope={slope:.4f}")
        return result


# =============================================================
# Complete Circle Experiment
# =============================================================

def run_circle_experiment(
    radius: float = 5.0,
    x0: float = 3.0,
    n_samples: int = 500,
    epochs: int = 500,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Run the complete circle/implicit function experiment.
    
    Steps:
    1. Generate circle dataset (upper semicircle)
    2. Train neural network to approximate the mapping
    3. Compute true tangent via implicit differentiation
    4. Compute AI tangent via autograd
    5. Compare results
    
    Args:
        radius: Circle radius.
        x0: Tangent point x-coordinate.
        n_samples: Number of training samples.
        epochs: Training epochs.
        device: Compute device.
    
    Returns:
        Dictionary with all experiment results.
    """
    from .model import CircleApproximator, get_device
    
    if device is None:
        device = get_device()
    
    logger.info(f"=" * 60)
    logger.info(f"CIRCLE EXPERIMENT: r={radius}, x0={x0}")
    logger.info(f"=" * 60)
    
    # 1. Initialize circle
    circle = CircleCurve(radius=radius)
    
    # 2. Generate dataset
    x_data, y_data = circle.generate_dataset(n_samples=n_samples, seed=42)
    
    # 3. Create and train model
    model = CircleApproximator(
        hidden_layers=[128, 128, 64, 32],
        activation='relu',
        radius=radius
    )
    
    trainer = CircleTrainer(model, device=device)
    history = trainer.train(
        x_data, y_data,
        epochs=epochs,
        lr=1e-3,
        batch_size=32,
        verbose=True,
        log_interval=100
    )
    
    # 4. Compute y0
    y0 = float(np.sqrt(radius ** 2 - x0 ** 2))
    
    # 5. Tangent computation
    tangent_x_range = np.linspace(x0 - 3, x0 + 3, 200).astype(np.float32)
    
    # True tangent (implicit differentiation)
    true_tangent = circle.tangent_at_point(x0, y0, x_range=tangent_x_range)
    
    # AI tangent
    ai_tangent = trainer.compute_ai_tangent(x0, x_range=tangent_x_range)
    
    # 6. Predictions for visualization
    x_plot = np.linspace(-radius + 0.01, radius - 0.01, 500).astype(np.float32)
    y_pred = trainer.predict(x_plot)
    y_true_plot = circle.y_from_x(x_plot, upper=True)
    
    # 7. Full circle points
    cx_full, cy_full = circle.generate_full_circle_points(1000)
    
    # 8. Compute metrics
    from .metrics import PerformanceMetrics
    func_mse = PerformanceMetrics.mse(y_true_plot, y_pred)
    func_r2 = PerformanceMetrics.r_squared(y_true_plot, y_pred)
    slope_err = abs(true_tangent['slope'] - ai_tangent['slope'])
    
    results = {
        'circle': circle,
        'model': model,
        'trainer': trainer,
        'history': history,
        'x_data': x_data,
        'y_data': y_data,
        'x_plot': x_plot,
        'y_pred': y_pred,
        'y_true_plot': y_true_plot,
        'x0': x0,
        'y0': y0,
        'true_tangent': true_tangent,
        'ai_tangent': ai_tangent,
        'circle_x_full': cx_full,
        'circle_y_full': cy_full,
        'metrics': {
            'function_mse': func_mse,
            'function_r_squared': func_r2,
            'slope_error': slope_err,
            'true_slope': true_tangent['slope'],
            'ai_slope': ai_tangent['slope'],
            'true_y0': y0,
            'ai_y0': ai_tangent['y0'],
        }
    }
    
    logger.info(f"Circle experiment complete. MSE={func_mse:.2e}, R²={func_r2:.6f}")
    return results
