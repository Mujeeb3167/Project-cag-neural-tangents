# =============================================================
# Professional Visualization Module
# AI-Based Neural Network System for Function Approximation
# Author: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)
# =============================================================

"""
Publication-quality visualization engine using matplotlib and seaborn.
All plots are exported in PNG (300 DPI) and PDF formats.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file export
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)

# =============================================================
# Professional Style Configuration
# =============================================================

PROFESSIONAL_STYLE = {
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.fontsize': 11,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'lines.linewidth': 2.2,
    'lines.markersize': 8,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
}

# Professional color palette
COLORS = {
    'primary': '#1976D2',
    'secondary': '#E53935',
    'accent': '#43A047',
    'warning': '#FB8C00',
    'dark': '#263238',
    'light': '#ECEFF1',
    'purple': '#8E24AA',
    'teal': '#00897B',
    'cyan': '#00ACC1',
    'pink': '#D81B60',
    'indigo': '#3949AB',
    'amber': '#FFB300',
}

PALETTE = [COLORS['primary'], COLORS['secondary'], COLORS['accent'],
           COLORS['purple'], COLORS['teal'], COLORS['warning']]


# =============================================================
# Visualization Engine
# =============================================================

class Visualizer:
    """
    Professional visualization engine for the AI function approximation project.
    
    Generates publication-quality plots with consistent styling.
    All figures are saved in both PNG (300 DPI) and PDF formats.
    
    Attributes:
        output_dir: Directory for saving figures.
    """
    
    def __init__(self, output_dir: str = 'figures') -> None:
        """
        Initialize the visualizer with professional styling.
        
        Args:
            output_dir: Directory to save generated figures.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._apply_style()
        logger.info(f"Visualizer initialized. Output dir: {output_dir}")
    
    def _apply_style(self) -> None:
        """Apply professional matplotlib styling."""
        plt.rcParams.update(PROFESSIONAL_STYLE)
        sns.set_context("notebook", font_scale=1.1)
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> Tuple[str, str]:
        """
        Save figure in both PNG and PDF formats.
        
        Args:
            fig: Matplotlib figure object.
            filename: Base filename (without extension).
        
        Returns:
            Tuple of (png_path, pdf_path).
        """
        png_path = os.path.join(self.output_dir, f"{filename}.png")
        pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        fig.savefig(pdf_path, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        
        logger.info(f"Figure saved: {png_path} and {pdf_path}")
        return png_path, pdf_path
    
    # ---------------------------------------------------------
    # Plot 1: Function Approximation
    # ---------------------------------------------------------
    
    def plot_function_approximation(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Neural Network Function Approximation',
        func_label: str = r'$f(x) = x^3 - 2x^2 + \sin(x)$',
        filename: str = 'function_approximation'
    ) -> str:
        """
        Plot original function vs neural network approximation with residual panel.
        
        Args:
            x: X-values.
            y_true: True function values.
            y_pred: Neural network predictions.
            title: Plot title.
            func_label: LaTeX label for the function.
            filename: Output filename.
        
        Returns:
            Path to saved PNG file.
        """
        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
        
        # --- Top panel: function comparison ---
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(x, y_true, color=COLORS['primary'], linewidth=2.5,
                 label=f'True: {func_label}', zorder=3)
        ax1.plot(x, y_pred, color=COLORS['secondary'], linewidth=2.0,
                 linestyle='--', label='Neural Network Approximation', zorder=2)
        ax1.fill_between(x, y_true, y_pred, alpha=0.1, color=COLORS['secondary'])
        ax1.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax1.set_ylabel('f(x)', fontsize=14)
        ax1.legend(loc='upper left', fontsize=12, framealpha=0.95)
        ax1.set_xticklabels([])
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # --- Bottom panel: residual ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        residual = y_true - y_pred
        ax2.fill_between(x, residual, 0, alpha=0.3, color=COLORS['accent'])
        ax2.plot(x, residual, color=COLORS['accent'], linewidth=1.5)
        ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax2.set_xlabel('x', fontsize=14)
        ax2.set_ylabel('Residual', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add MSE annotation
        mse_val = float(np.mean(residual ** 2))
        ax2.text(0.98, 0.85, f'MSE = {mse_val:.2e}',
                 transform=ax2.transAxes, fontsize=11,
                 ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLORS['accent'], alpha=0.9))
        
        png_path, _ = self._save_figure(fig, filename)
        return png_path
    
    # ---------------------------------------------------------
    # Plot 2: Tangent Line Comparison
    # ---------------------------------------------------------
    
    def plot_tangent_comparison(
        self,
        x: np.ndarray,
        y_func: np.ndarray,
        x0: float,
        true_tangent: Dict,
        ai_tangent: Dict,
        filename: str = 'tangent_comparison'
    ) -> str:
        """
        Plot true vs AI tangent lines at a specified point.
        
        Args:
            x: X-values for the function.
            y_func: Function values.
            x0: Tangent point x-coordinate.
            true_tangent: True tangent dictionary.
            ai_tangent: AI tangent dictionary.
            filename: Output filename.
        
        Returns:
            Path to saved PNG file.
        """
        fig, ax = plt.subplots(figsize=(13, 8))
        
        # Plot function
        ax.plot(x, y_func, color=COLORS['dark'], linewidth=2.5,
                label=r'$f(x) = x^3 - 2x^2 + \sin(x)$', zorder=2)
        
        # Tangent line range (zoomed around x0)
        tangent_x = true_tangent.get('x_range', np.linspace(x0 - 2, x0 + 2, 100))
        
        # True tangent
        if true_tangent.get('y_tangent') is not None:
            ax.plot(tangent_x, true_tangent['y_tangent'],
                    color=COLORS['primary'], linewidth=2.5, linestyle='-',
                    label=f"True Tangent: {true_tangent['equation']}", zorder=3)
        
        # AI tangent
        if ai_tangent.get('y_tangent') is not None:
            ax.plot(tangent_x, ai_tangent['y_tangent'],
                    color=COLORS['secondary'], linewidth=2.5, linestyle='--',
                    label=f"AI Tangent: {ai_tangent['equation']}", zorder=3)
        
        # Tangent point marker
        f_x0 = true_tangent['f_x0']
        ax.scatter([x0], [f_x0], color=COLORS['warning'], s=150,
                   zorder=5, edgecolors='black', linewidths=1.5,
                   label=f'Tangent Point ({x0:.2f}, {f_x0:.2f})')
        
        # Annotations
        ax.annotate(f'x₀ = {x0}',
                    xy=(x0, f_x0), xytext=(x0 + 0.5, f_x0 + 3),
                    fontsize=12, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                   connectionstyle='arc3,rad=0.2'),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=COLORS['warning'], alpha=0.95))
        
        ax.set_title('Tangent Line Comparison: True vs AI-Derived',
                     fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        png_path, _ = self._save_figure(fig, filename)
        return png_path
    
    # ---------------------------------------------------------
    # Plot 3: Training Loss Curve
    # ---------------------------------------------------------
    
    def plot_training_loss(
        self,
        history: Dict[str, List[float]],
        filename: str = 'training_loss'
    ) -> str:
        """
        Plot training and validation loss curves with log scale.
        
        Args:
            history: Training history with 'train_loss' and 'val_loss'.
            filename: Output filename.
        
        Returns:
            Path to saved PNG file.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        epochs = range(1, len(history['train_loss']) + 1)
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        
        # --- Left: Log-scale loss ---
        ax1.semilogy(epochs, train_loss, color=COLORS['primary'],
                     linewidth=2.0, label='Training Loss', alpha=0.9)
        ax1.semilogy(epochs, val_loss, color=COLORS['secondary'],
                     linewidth=2.0, label='Validation Loss', alpha=0.9)
        
        # Mark best epoch
        best_epoch = int(np.argmin(val_loss)) + 1
        best_val = min(val_loss)
        ax1.axvline(x=best_epoch, color=COLORS['accent'], linestyle=':',
                    linewidth=1.5, alpha=0.7)
        ax1.scatter([best_epoch], [best_val], color=COLORS['accent'],
                    s=100, zorder=5, edgecolors='black',
                    label=f'Best (epoch {best_epoch})')
        
        ax1.set_title('Training Progress (Log Scale)',
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=13)
        ax1.set_ylabel('Loss (log scale)', fontsize=13)
        ax1.legend(fontsize=11, framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # --- Right: Learning rate ---
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            ax2.semilogy(epochs, history['learning_rate'],
                        color=COLORS['purple'], linewidth=2.0)
            ax2.set_title('Learning Rate Schedule',
                         fontsize=16, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=13)
            ax2.set_ylabel('Learning Rate', fontsize=13)
            ax2.grid(True, alpha=0.3, linestyle='--')
        else:
            # If no LR data, show linear-scale loss
            ax2.plot(epochs, train_loss, color=COLORS['primary'],
                     linewidth=2.0, label='Training Loss')
            ax2.plot(epochs, val_loss, color=COLORS['secondary'],
                     linewidth=2.0, label='Validation Loss')
            ax2.set_title('Training Progress (Linear Scale)',
                         fontsize=16, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=13)
            ax2.set_ylabel('Loss', fontsize=13)
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3, linestyle='--')
        
        fig.suptitle('Neural Network Training Analysis',
                    fontsize=20, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        png_path, _ = self._save_figure(fig, filename)
        return png_path
    
    # ---------------------------------------------------------
    # Plot 4: Derivative Comparison
    # ---------------------------------------------------------
    
    def plot_derivative_comparison(
        self,
        x: np.ndarray,
        true_deriv: np.ndarray,
        auto_diff_deriv: np.ndarray,
        finite_diff_deriv: np.ndarray,
        filename: str = 'derivative_comparison'
    ) -> str:
        """
        Plot comparison of derivatives from different methods.
        
        Args:
            x: X-values.
            true_deriv: True analytic derivative.
            auto_diff_deriv: Autograd derivative.
            finite_diff_deriv: Finite difference derivative.
            filename: Output filename.
        
        Returns:
            Path to saved PNG file.
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.12)
        
        # --- Top: Derivative overlay ---
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(x, true_deriv, color=COLORS['dark'], linewidth=2.5,
                 label=r"True: $f'(x) = 3x^2 - 4x + \cos(x)$", zorder=3)
        ax1.plot(x, auto_diff_deriv, color=COLORS['primary'],
                 linewidth=2.0, linestyle='--',
                 label='AutoDiff (PyTorch)', zorder=2)
        ax1.plot(x, finite_diff_deriv, color=COLORS['secondary'],
                 linewidth=2.0, linestyle=':',
                 label='Finite Difference', zorder=2)
        
        ax1.set_title('Derivative Comparison: True vs AutoDiff vs Finite Difference',
                      fontsize=16, fontweight='bold', pad=12)
        ax1.set_ylabel("f'(x)", fontsize=14)
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax1.set_xticklabels([])
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # --- Bottom: Errors ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ad_error = np.abs(true_deriv - auto_diff_deriv)
        fd_error = np.abs(true_deriv - finite_diff_deriv)
        
        ax2.semilogy(x, ad_error + 1e-15, color=COLORS['primary'],
                     linewidth=1.5, label='AutoDiff Error')
        ax2.semilogy(x, fd_error + 1e-15, color=COLORS['secondary'],
                     linewidth=1.5, label='Finite Diff Error')
        
        ax2.set_xlabel('x', fontsize=14)
        ax2.set_ylabel('|Error| (log)', fontsize=12)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        png_path, _ = self._save_figure(fig, filename)
        return png_path
    
    # ---------------------------------------------------------
    # Plot 5: Error Analysis
    # ---------------------------------------------------------
    
    def plot_error_analysis(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        true_deriv: np.ndarray,
        ai_deriv: np.ndarray,
        filename: str = 'error_analysis'
    ) -> str:
        """
        Plot error analysis with histograms and spatial distribution.
        
        Args:
            x: X-values.
            y_true: True function values.
            y_pred: Predicted function values.
            true_deriv: True derivative values.
            ai_deriv: AI derivative values.
            filename: Output filename.
        
        Returns:
            Path to saved PNG file.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        
        func_error = y_true - y_pred
        deriv_error = true_deriv - ai_deriv
        
        # (0,0) Function error distribution
        axes[0, 0].hist(func_error, bins=40, color=COLORS['primary'],
                        alpha=0.7, edgecolor='white', linewidth=0.5)
        axes[0, 0].axvline(x=0, color='black', linewidth=1, linestyle='--')
        axes[0, 0].set_title('Function Approximation Error Distribution',
                            fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Error', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        
        # (0,1) Derivative error distribution
        axes[0, 1].hist(deriv_error, bins=40, color=COLORS['secondary'],
                        alpha=0.7, edgecolor='white', linewidth=0.5)
        axes[0, 1].axvline(x=0, color='black', linewidth=1, linestyle='--')
        axes[0, 1].set_title('Derivative Error Distribution',
                            fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Error', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        
        # (1,0) Spatial error: function
        axes[1, 0].scatter(x, np.abs(func_error), color=COLORS['primary'],
                          alpha=0.5, s=15, zorder=2)
        axes[1, 0].plot(x, np.abs(func_error), color=COLORS['primary'],
                       alpha=0.3, linewidth=0.8)
        axes[1, 0].set_title('Function Error vs Position',
                            fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('x', fontsize=12)
        axes[1, 0].set_ylabel('|Error|', fontsize=12)
        axes[1, 0].set_yscale('log')
        
        # (1,1) Spatial error: derivative
        axes[1, 1].scatter(x, np.abs(deriv_error), color=COLORS['secondary'],
                          alpha=0.5, s=15, zorder=2)
        axes[1, 1].plot(x, np.abs(deriv_error), color=COLORS['secondary'],
                       alpha=0.3, linewidth=0.8)
        axes[1, 1].set_title('Derivative Error vs Position',
                            fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('x', fontsize=12)
        axes[1, 1].set_ylabel('|Error|', fontsize=12)
        axes[1, 1].set_yscale('log')
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.suptitle('Comprehensive Error Analysis',
                    fontsize=18, fontweight='bold', y=1.01)
        fig.tight_layout()
        
        png_path, _ = self._save_figure(fig, filename)
        return png_path
    
    # ---------------------------------------------------------
    # Plot 6: Circle Visualization
    # ---------------------------------------------------------
    
    def plot_circle(
        self,
        radius: float = 5.0,
        x0: float = 3.0,
        y0: Optional[float] = None,
        tangent_data: Optional[Dict] = None,
        ai_tangent_data: Optional[Dict] = None,
        nn_x: Optional[np.ndarray] = None,
        nn_y: Optional[np.ndarray] = None,
        filename: str = 'circle_visualization'
    ) -> str:
        """
        Plot circle with tangent lines and NN approximation.
        
        Args:
            radius: Circle radius.
            x0: Tangent point x-coordinate.
            y0: Tangent point y-coordinate (computed if None).
            tangent_data: True tangent line data.
            ai_tangent_data: AI tangent line data.
            nn_x: NN-predicted x values.
            nn_y: NN-predicted y values.
            filename: Output filename.
        
        Returns:
            Path to saved PNG file.
        """
        if y0 is None:
            y0 = float(np.sqrt(radius ** 2 - x0 ** 2))
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw circle
        theta = np.linspace(0, 2 * np.pi, 500)
        cx = radius * np.cos(theta)
        cy = radius * np.sin(theta)
        ax.plot(cx, cy, color=COLORS['dark'], linewidth=2.5,
                label=f'Circle: x² + y² = {radius**2:.0f}', zorder=2)
        
        # NN approximation of upper semicircle
        if nn_x is not None and nn_y is not None:
            ax.plot(nn_x, nn_y, color=COLORS['primary'], linewidth=2.0,
                    linestyle='--', label='NN Approximation', zorder=3)
        
        # True tangent line
        if tangent_data and tangent_data.get('y_tangent') is not None:
            ax.plot(tangent_data['x_range'], tangent_data['y_tangent'],
                    color=COLORS['accent'], linewidth=2.5,
                    label=f"True Tangent: {tangent_data['equation']}", zorder=3)
        
        # AI tangent line
        if ai_tangent_data and ai_tangent_data.get('y_tangent') is not None:
            ax.plot(ai_tangent_data['x_range'], ai_tangent_data['y_tangent'],
                    color=COLORS['secondary'], linewidth=2.5, linestyle='--',
                    label=f"AI Tangent: {ai_tangent_data['equation']}", zorder=3)
        
        # Tangent point
        ax.scatter([x0], [y0], color=COLORS['warning'], s=150,
                   zorder=5, edgecolors='black', linewidths=1.5,
                   label=f'({x0:.1f}, {y0:.2f})')
        
        # Center point
        ax.scatter([0], [0], color=COLORS['dark'], s=80,
                   marker='+', linewidths=2, zorder=5)
        
        # Axes
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='-')
        
        ax.set_title(f'Circle x² + y² = {radius**2:.0f} with Tangent Lines',
                     fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_aspect('equal')
        ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-radius - 2, radius + 2)
        ax.set_ylim(-radius - 2, radius + 2)
        
        png_path, _ = self._save_figure(fig, filename)
        return png_path
    
    # ---------------------------------------------------------
    # Plot 7: Implicit Tangent Line
    # ---------------------------------------------------------
    
    def plot_implicit_tangent(
        self,
        radius: float = 5.0,
        x0: float = 3.0,
        y0: Optional[float] = None,
        true_tangent: Optional[Dict] = None,
        ai_tangent: Optional[Dict] = None,
        filename: str = 'implicit_tangent'
    ) -> str:
        """
        Plot implicit tangent line on circle (zoomed view).
        
        Args:
            radius: Circle radius.
            x0: Tangent point x-coordinate.
            y0: Tangent point y-coordinate.
            true_tangent: True tangent data.
            ai_tangent: AI tangent data.
            filename: Output filename.
        
        Returns:
            Path to saved PNG file.
        """
        if y0 is None:
            y0 = float(np.sqrt(radius ** 2 - x0 ** 2))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Upper semicircle (zoomed)
        x_semi = np.linspace(-radius, radius, 500)
        y_semi = np.sqrt(np.maximum(radius ** 2 - x_semi ** 2, 0))
        ax.plot(x_semi, y_semi, color=COLORS['dark'], linewidth=2.5,
                label='Upper Semicircle', zorder=2)
        
        # True tangent
        if true_tangent and true_tangent.get('y_tangent') is not None:
            ax.plot(true_tangent['x_range'], true_tangent['y_tangent'],
                    color=COLORS['accent'], linewidth=2.5,
                    label=f"True: {true_tangent['equation']}", zorder=3)
        
        # AI tangent
        if ai_tangent and ai_tangent.get('y_tangent') is not None:
            ax.plot(ai_tangent['x_range'], ai_tangent['y_tangent'],
                    color=COLORS['secondary'], linewidth=2.5, linestyle='--',
                    label=f"AI: {ai_tangent['equation']}", zorder=3)
        
        # Tangent point
        ax.scatter([x0], [y0], color=COLORS['warning'], s=180,
                   zorder=5, edgecolors='black', linewidths=2,
                   label=f'({x0:.1f}, {y0:.2f})')
        
        # Annotation
        ax.annotate(
            f'dy/dx = −x/y = {-x0/y0:.4f}',
            xy=(x0, y0), xytext=(x0 + 1.5, y0 + 1),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['dark']),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                     edgecolor=COLORS['warning'], alpha=0.95)
        )
        
        ax.set_title('Implicit Differentiation: Tangent to Circle',
                     fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        png_path, _ = self._save_figure(fig, filename)
        return png_path
    
    # ---------------------------------------------------------
    # Plot 8: Noise Comparison
    # ---------------------------------------------------------
    
    def plot_noise_comparison(
        self,
        x: np.ndarray,
        y_clean: np.ndarray,
        y_noisy: np.ndarray,
        y_pred_clean: np.ndarray,
        y_pred_noisy: np.ndarray,
        filename: str = 'noise_comparison'
    ) -> str:
        """
        Plot noise vs no-noise comparison in a 2x2 grid.
        
        Args:
            x: X-values.
            y_clean: Clean function values.
            y_noisy: Noisy function values.
            y_pred_clean: NN predictions (trained on clean data).
            y_pred_noisy: NN predictions (trained on noisy data).
            filename: Output filename.
        
        Returns:
            Path to saved PNG file.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # (0,0) Clean data + fit
        axes[0, 0].plot(x, y_clean, color=COLORS['primary'], linewidth=2.0,
                        label='True Function')
        axes[0, 0].plot(x, y_pred_clean, color=COLORS['secondary'],
                        linewidth=2.0, linestyle='--', label='NN (Clean)')
        axes[0, 0].set_title('Clean Data: Function vs NN',
                            fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        
        # (0,1) Noisy data + fit
        axes[0, 1].scatter(x, y_noisy, color=COLORS['primary'],
                          alpha=0.3, s=8, label='Noisy Data')
        axes[0, 1].plot(x, y_pred_noisy, color=COLORS['secondary'],
                        linewidth=2.0, label='NN (Noisy)')
        axes[0, 1].set_title('Noisy Data: Data vs NN',
                            fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        
        # (1,0) Clean residual
        clean_err = np.abs(y_clean - y_pred_clean)
        axes[1, 0].fill_between(x, clean_err, 0, alpha=0.3, color=COLORS['accent'])
        axes[1, 0].plot(x, clean_err, color=COLORS['accent'], linewidth=1.5)
        mse_clean = float(np.mean((y_clean - y_pred_clean) ** 2))
        axes[1, 0].set_title(f'Clean Error (MSE={mse_clean:.2e})',
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('x', fontsize=12)
        
        # (1,1) Noisy residual
        # Compare noisy prediction against TRUE function (not noisy data)
        noisy_err = np.abs(y_clean - y_pred_noisy)
        axes[1, 1].fill_between(x, noisy_err, 0, alpha=0.3, color=COLORS['warning'])
        axes[1, 1].plot(x, noisy_err, color=COLORS['warning'], linewidth=1.5)
        mse_noisy = float(np.mean((y_clean - y_pred_noisy) ** 2))
        axes[1, 1].set_title(f'Noisy Model Error (MSE={mse_noisy:.2e})',
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('x', fontsize=12)
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylabel('y' if ax in axes[:, 0] else '', fontsize=12)
        
        fig.suptitle('Impact of Noise on Neural Network Approximation',
                    fontsize=18, fontweight='bold', y=1.01)
        fig.tight_layout()
        
        png_path, _ = self._save_figure(fig, filename)
        return png_path
    
    # ---------------------------------------------------------
    # Metrics Dashboard
    # ---------------------------------------------------------
    
    def plot_metrics_dashboard(
        self,
        metrics: Dict,
        filename: str = 'metrics_dashboard'
    ) -> str:
        """
        Create a visual metrics dashboard.
        
        Args:
            metrics: Dictionary of metric values.
            filename: Output filename.
        
        Returns:
            Path to saved PNG file.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Group metrics
        func_metrics = {k: v for k, v in metrics.items()
                       if k.startswith('function_') and isinstance(v, (int, float))}
        deriv_metrics = {k: v for k, v in metrics.items()
                        if k.startswith('derivative_') and isinstance(v, (int, float))}
        tangent_metrics = {k: v for k, v in metrics.items()
                          if any(k.startswith(p) for p in ['slope_', 'intercept_', 'tangent_'])
                          and isinstance(v, (int, float))}
        
        def _plot_bar_group(ax, data, title, color):
            if not data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(title, fontsize=14, fontweight='bold')
                return
            names = [k.split('_', 1)[-1].replace('_', ' ').title()
                     for k in data.keys()]
            values = list(data.values())
            bars = ax.barh(names, values, color=color, alpha=0.8,
                          edgecolor='white', linewidth=0.5)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Value', fontsize=11)
            # Add value labels
            for bar, val in zip(bars, values):
                label = f'{val:.2e}' if abs(val) < 0.001 or abs(val) > 1000 else f'{val:.4f}'
                ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                        label, va='center', fontsize=9)
        
        _plot_bar_group(axes[0], func_metrics, 'Function Approximation', COLORS['primary'])
        _plot_bar_group(axes[1], deriv_metrics, 'Derivative Estimation', COLORS['secondary'])
        _plot_bar_group(axes[2], tangent_metrics, 'Tangent Line', COLORS['accent'])
        
        for ax in axes:
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        fig.suptitle('Performance Metrics Dashboard',
                    fontsize=18, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        png_path, _ = self._save_figure(fig, filename)
        return png_path
    
    # ---------------------------------------------------------
    # PINN Comparison
    # ---------------------------------------------------------
    
    def plot_pinn_comparison(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred_nn: np.ndarray,
        y_pred_pinn: np.ndarray,
        history_nn: Dict,
        history_pinn: Dict,
        filename: str = 'pinn_comparison'
    ) -> str:
        """
        Compare standard NN vs Physics-Informed NN.
        
        Args:
            x: X-values.
            y_true: True function values.
            y_pred_nn: Standard NN predictions.
            y_pred_pinn: PINN predictions.
            history_nn: Standard NN training history.
            history_pinn: PINN training history.
            filename: Output filename.
        
        Returns:
            Path to saved PNG file.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # (0) Function comparison
        axes[0].plot(x, y_true, color=COLORS['dark'], linewidth=2.5,
                    label='True Function', zorder=3)
        axes[0].plot(x, y_pred_nn, color=COLORS['primary'], linewidth=2.0,
                    linestyle='--', label='Standard NN')
        axes[0].plot(x, y_pred_pinn, color=COLORS['accent'], linewidth=2.0,
                    linestyle=':', label='PINN')
        axes[0].set_title('Function Approximation', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('f(x)')
        
        # (1) Error comparison
        nn_err = np.abs(y_true - y_pred_nn)
        pinn_err = np.abs(y_true - y_pred_pinn)
        axes[1].semilogy(x, nn_err + 1e-15, color=COLORS['primary'],
                        linewidth=1.5, label='Standard NN Error')
        axes[1].semilogy(x, pinn_err + 1e-15, color=COLORS['accent'],
                        linewidth=1.5, label='PINN Error')
        axes[1].set_title('Approximation Error', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('|Error| (log)')
        
        # (2) Loss comparison
        axes[2].semilogy(history_nn['val_loss'], color=COLORS['primary'],
                        linewidth=1.5, label='NN Val Loss')
        axes[2].semilogy(history_pinn['val_loss'], color=COLORS['accent'],
                        linewidth=1.5, label='PINN Val Loss')
        axes[2].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss (log)')
        
        for ax in axes:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.suptitle('Standard Neural Network vs Physics-Informed Neural Network',
                    fontsize=17, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        png_path, _ = self._save_figure(fig, filename)
        return png_path
    
    # ---------------------------------------------------------
    # Master Plot Generator
    # ---------------------------------------------------------
    
    def create_all_plots(self, results: Dict) -> Dict[str, str]:
        """
        Generate all visualization plots from a results dictionary.
        
        Args:
            results: Dictionary containing all data needed for plotting.
        
        Returns:
            Dictionary mapping plot names to file paths.
        """
        paths = {}
        
        try:
            if all(k in results for k in ['x', 'y_true', 'y_pred']):
                paths['function_approximation'] = self.plot_function_approximation(
                    results['x'], results['y_true'], results['y_pred']
                )
        except Exception as e:
            logger.error(f"Error in function approximation plot: {e}")
        
        try:
            if all(k in results for k in ['x', 'y_true', 'x0', 'true_tangent', 'ai_tangent']):
                paths['tangent_comparison'] = self.plot_tangent_comparison(
                    results['x'], results['y_true'], results['x0'],
                    results['true_tangent'], results['ai_tangent']
                )
        except Exception as e:
            logger.error(f"Error in tangent comparison plot: {e}")
        
        try:
            if 'history' in results:
                paths['training_loss'] = self.plot_training_loss(results['history'])
        except Exception as e:
            logger.error(f"Error in training loss plot: {e}")
        
        try:
            if all(k in results for k in ['x', 'true_deriv', 'auto_diff_deriv', 'finite_diff_deriv']):
                paths['derivative_comparison'] = self.plot_derivative_comparison(
                    results['x'], results['true_deriv'],
                    results['auto_diff_deriv'], results['finite_diff_deriv']
                )
        except Exception as e:
            logger.error(f"Error in derivative comparison plot: {e}")
        
        try:
            if all(k in results for k in ['x', 'y_true', 'y_pred', 'true_deriv', 'auto_diff_deriv']):
                paths['error_analysis'] = self.plot_error_analysis(
                    results['x'], results['y_true'], results['y_pred'],
                    results['true_deriv'], results['auto_diff_deriv']
                )
        except Exception as e:
            logger.error(f"Error in error analysis plot: {e}")
        
        try:
            if 'metrics' in results:
                paths['metrics_dashboard'] = self.plot_metrics_dashboard(results['metrics'])
        except Exception as e:
            logger.error(f"Error in metrics dashboard plot: {e}")
        
        logger.info(f"Generated {len(paths)} plots successfully.")
        return paths
