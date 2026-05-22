# =============================================================
# Training System Module
# AI-Based Neural Network System for Function Approximation
# Author: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)
# =============================================================

"""
Professional training system with Adam optimizer, learning rate
scheduling, early stopping, and training history tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import time
import copy
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================
# Early Stopping
# =============================================================

class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    
    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs (patience).
    Optionally restores the best model weights.
    
    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        counter: Current number of epochs without improvement.
        best_loss: Best validation loss observed so far.
        should_stop: Flag indicating whether training should stop.
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-6,
        restore_best: bool = True
    ) -> None:
        """
        Initialize early stopping.
        
        Args:
            patience: Epochs to wait before stopping.
            min_delta: Minimum improvement threshold.
            restore_best: Whether to restore best model weights on stop.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter: int = 0
        self.best_loss: Optional[float] = None
        self.best_model_state: Optional[dict] = None
        self.should_stop: bool = False
        self.best_epoch: int = 0
    
    def __call__(self, val_loss: float, model: nn.Module, epoch: int = 0) -> None:
        """
        Check whether training should stop.
        
        Args:
            val_loss: Current epoch's validation loss.
            model: The model being trained (for saving best state).
            epoch: Current epoch number.
        """
        if self.best_loss is None:
            # First call — initialize
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            # Improvement detected
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best loss: {self.best_loss:.6f} at epoch {self.best_epoch}"
                )
    
    def restore(self, model: nn.Module) -> None:
        """
        Restore the best model weights.
        
        Args:
            model: The model to restore weights into.
        """
        if self.restore_best and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model from epoch {self.best_epoch}")


# =============================================================
# Standard Trainer
# =============================================================

class Trainer:
    """
    Professional training manager for neural networks.
    
    Handles the complete training pipeline including:
    - Adam optimizer with configurable parameters
    - Learning rate scheduling (ReduceLROnPlateau or CosineAnnealing)
    - Early stopping
    - Training/validation loop
    - Loss and metric history tracking
    - Model checkpointing
    
    Attributes:
        model: The neural network to train.
        optimizer: Adam optimizer instance.
        scheduler: Learning rate scheduler.
        criterion: Loss function (MSELoss).
        device: Compute device (CPU/GPU).
        history: Dictionary tracking training metrics over epochs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[torch.device] = None,
        scheduler_type: str = 'reduce_on_plateau',
        early_stopping_patience: int = 50
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model to train.
            learning_rate: Initial learning rate for Adam.
            weight_decay: L2 regularization strength.
            device: Device for training. Auto-detects if None.
            scheduler_type: LR scheduler type ('reduce_on_plateau' or 'cosine').
            early_stopping_patience: Patience for early stopping.
        """
        from .model import get_device
        
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        
        # Optimizer: Adam with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=20,
                min_lr=1e-7
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-7
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        self.scheduler_type = scheduler_type
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            restore_best=True
        )
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        logger.info(
            f"Trainer initialized: lr={learning_rate}, device={self.device}, "
            f"scheduler={scheduler_type}, early_stop_patience={early_stopping_patience}"
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Execute one training epoch.
        
        Args:
            train_loader: DataLoader for training data.
        
        Returns:
            Average training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(x_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Execute validation pass.
        
        Args:
            val_loader: DataLoader for validation data.
        
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(x_batch)
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 1000,
        verbose: bool = True,
        log_interval: int = 100
    ) -> Dict[str, List[float]]:
        """
        Execute the full training loop.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Maximum number of training epochs.
            verbose: Whether to print progress logs.
            log_interval: Print progress every N epochs.
        
        Returns:
            Training history dictionary with 'train_loss', 'val_loss',
            and 'learning_rate' lists.
        """
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Training step
            train_loss = self.train_epoch(train_loader)
            
            # Validation step
            val_loss = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Update learning rate scheduler
            if self.scheduler_type == 'reduce_on_plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Check early stopping
            self.early_stopping(val_loss, self.model, epoch)
            if self.early_stopping.should_stop:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if verbose and (epoch % log_interval == 0 or epoch == 1):
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
        
        # Restore best model
        self.early_stopping.restore(self.model)
        
        total_time = time.time() - start_time
        best_val = min(self.history['val_loss'])
        logger.info(
            f"Training complete in {total_time:.1f}s. "
            f"Best val loss: {best_val:.6f} at epoch {self.early_stopping.best_epoch}"
        )
        
        return self.history
    
    def get_history(self) -> Dict[str, List[float]]:
        """Return the training history dictionary."""
        return self.history


# =============================================================
# Physics-Informed Neural Network Trainer
# =============================================================

class PINNTrainer(Trainer):
    """
    Trainer for Physics-Informed Neural Networks.
    
    Extends the standard Trainer by adding a physics loss term
    that penalizes deviation of the network's derivative from
    the known analytic derivative.
    
    Total loss = data_loss + physics_weight * physics_loss
    
    Attributes:
        true_derivative_func: Callable returning the true derivative.
        physics_weight: Weight for the physics loss term.
    """
    
    def __init__(
        self,
        model: nn.Module,
        true_derivative_func: Callable,
        physics_weight: float = 0.1,
        **kwargs
    ) -> None:
        """
        Initialize the PINN trainer.
        
        Args:
            model: The PINN model.
            true_derivative_func: Function returning true derivative values (torch tensor).
            physics_weight: Weight for the derivative constraint loss.
            **kwargs: Additional arguments passed to base Trainer.
        """
        super().__init__(model, **kwargs)
        self.true_derivative_func = true_derivative_func
        self.physics_weight = physics_weight
        logger.info(f"PINNTrainer initialized with physics_weight={physics_weight}")
    
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the physics-informed loss (derivative constraint).
        
        Uses autograd to compute the network's derivative and compares
        it with the known true derivative.
        
        Args:
            x: Input tensor with requires_grad=True.
        
        Returns:
            MSE between network derivative and true derivative.
        """
        x_grad = x.clone().detach().requires_grad_(True)
        y_pred = self.model(x_grad)
        
        # Compute dy/dx using autograd
        dy_dx = torch.autograd.grad(
            outputs=y_pred,
            inputs=x_grad,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # True derivative values
        true_deriv = self.true_derivative_func(x_grad)
        
        # Physics loss: MSE between predicted and true derivatives
        return self.criterion(dy_dx, true_deriv)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Execute one training epoch with physics-informed loss.
        
        Total loss = data_loss + physics_weight * physics_loss
        
        Args:
            train_loader: DataLoader for training data.
        
        Returns:
            Average total loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Data loss
            predictions = self.model(x_batch)
            data_loss = self.criterion(predictions, y_batch)
            
            # Physics loss (derivative constraint)
            phys_loss = self.physics_loss(x_batch)
            
            # Combined loss
            loss = data_loss + self.physics_weight * phys_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
