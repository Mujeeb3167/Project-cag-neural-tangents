# =============================================================
# Performance Metrics Module
# AI-Based Neural Network System for Function Approximation
# Author: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)
# =============================================================

"""
Comprehensive performance metrics for evaluating function approximation,
derivative estimation, and tangent line accuracy.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from tabulate import tabulate
import json
import os
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator and formatter.
    
    Supports evaluation of:
    - Function approximation accuracy (MSE, MAE, RMSE, R²)
    - Tangent line accuracy (slope error, intercept error, tangent MSE)
    - Derivative estimation accuracy
    
    All methods are static for maximum flexibility.
    """
    
    # ---------------------------------------------------------
    # Core Regression Metrics
    # ---------------------------------------------------------
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error.
        
        MSE = (1/n) Σ (y_true - y_pred)²
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
        
        Returns:
            MSE value as a float.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        return float(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Absolute Error.
        
        MAE = (1/n) Σ |y_true - y_pred|
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
        
        Returns:
            MAE value as a float.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Root Mean Squared Error.
        
        RMSE = sqrt(MSE)
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
        
        Returns:
            RMSE value as a float.
        """
        return float(np.sqrt(PerformanceMetrics.mse(y_true, y_pred)))
    
    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Coefficient of Determination (R²).
        
        R² = 1 - SS_res / SS_tot
        where SS_res = Σ(y_true - y_pred)², SS_tot = Σ(y_true - ȳ)²
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
        
        Returns:
            R² value (1.0 = perfect, 0.0 = as good as mean).
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot < 1e-15:
            logger.warning("R² undefined: constant target values (SS_tot ≈ 0)")
            return 1.0 if ss_res < 1e-15 else 0.0
        
        return float(1.0 - ss_res / ss_tot)
    
    @staticmethod
    def max_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Maximum Absolute Error.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
        
        Returns:
            Maximum absolute error as a float.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        return float(np.max(np.abs(y_true - y_pred)))
    
    # ---------------------------------------------------------
    # Tangent Line Metrics
    # ---------------------------------------------------------
    
    @staticmethod
    def slope_error(true_slope: float, ai_slope: float) -> float:
        """
        Compute absolute error between true and AI tangent slopes.
        
        Args:
            true_slope: True analytic slope.
            ai_slope: AI-derived slope.
        
        Returns:
            Absolute slope error.
        """
        return float(abs(true_slope - ai_slope))
    
    @staticmethod
    def intercept_error(true_intercept: float, ai_intercept: float) -> float:
        """
        Compute absolute error between true and AI tangent intercepts.
        
        Args:
            true_intercept: True intercept.
            ai_intercept: AI-derived intercept.
        
        Returns:
            Absolute intercept error.
        """
        return float(abs(true_intercept - ai_intercept))
    
    @staticmethod
    def tangent_line_mse(
        true_tangent_y: np.ndarray,
        ai_tangent_y: np.ndarray
    ) -> float:
        """
        Compute MSE between true and AI tangent line y-values.
        
        Args:
            true_tangent_y: Y-values of the true tangent line.
            ai_tangent_y: Y-values of the AI tangent line.
        
        Returns:
            Tangent line MSE.
        """
        return PerformanceMetrics.mse(true_tangent_y, ai_tangent_y)
    
    # ---------------------------------------------------------
    # Derivative Metrics
    # ---------------------------------------------------------
    
    @staticmethod
    def derivative_error(
        true_deriv: np.ndarray,
        ai_deriv: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive derivative error metrics.
        
        Args:
            true_deriv: True analytic derivative values.
            ai_deriv: AI-computed derivative values.
        
        Returns:
            Dictionary with MSE, MAE, RMSE, R², and max error.
        """
        true_deriv = np.asarray(true_deriv).flatten()
        ai_deriv = np.asarray(ai_deriv).flatten()
        
        return {
            'derivative_mse': PerformanceMetrics.mse(true_deriv, ai_deriv),
            'derivative_mae': PerformanceMetrics.mae(true_deriv, ai_deriv),
            'derivative_rmse': PerformanceMetrics.rmse(true_deriv, ai_deriv),
            'derivative_r_squared': PerformanceMetrics.r_squared(true_deriv, ai_deriv),
            'derivative_max_error': PerformanceMetrics.max_absolute_error(true_deriv, ai_deriv)
        }
    
    # ---------------------------------------------------------
    # Comprehensive Metrics
    # ---------------------------------------------------------
    
    @classmethod
    def compute_all_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        true_tangent: Dict,
        ai_tangent: Dict,
        true_deriv: np.ndarray,
        ai_deriv: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute ALL performance metrics in one call.
        
        Args:
            y_true: True function values.
            y_pred: NN-predicted function values.
            true_tangent: True tangent line dictionary.
            ai_tangent: AI tangent line dictionary.
            true_deriv: True derivative values.
            ai_deriv: AI derivative values.
        
        Returns:
            Comprehensive dictionary of all metrics.
        """
        metrics = {}
        
        # Function approximation metrics
        metrics['function_mse'] = cls.mse(y_true, y_pred)
        metrics['function_mae'] = cls.mae(y_true, y_pred)
        metrics['function_rmse'] = cls.rmse(y_true, y_pred)
        metrics['function_r_squared'] = cls.r_squared(y_true, y_pred)
        metrics['function_max_error'] = cls.max_absolute_error(y_true, y_pred)
        
        # Tangent line metrics
        metrics['slope_error'] = cls.slope_error(
            true_tangent['slope'], ai_tangent['slope']
        )
        metrics['intercept_error'] = cls.intercept_error(
            true_tangent['intercept'], ai_tangent['intercept']
        )
        
        if (true_tangent.get('y_tangent') is not None and
                ai_tangent.get('y_tangent') is not None):
            metrics['tangent_line_mse'] = cls.tangent_line_mse(
                true_tangent['y_tangent'], ai_tangent['y_tangent']
            )
        
        # Derivative metrics
        deriv_metrics = cls.derivative_error(true_deriv, ai_deriv)
        metrics.update(deriv_metrics)
        
        logger.info(f"All metrics computed: {len(metrics)} metrics total")
        return metrics
    
    # ---------------------------------------------------------
    # Display and Formatting
    # ---------------------------------------------------------
    
    @staticmethod
    def format_metrics_table(
        metrics: Dict[str, float],
        title: str = "Performance Metrics"
    ) -> str:
        """
        Format metrics as a clean, professional table.
        
        Args:
            metrics: Dictionary of metric name → value.
            title: Table title.
        
        Returns:
            Formatted table string.
        """
        # Human-readable metric names
        name_map = {
            'function_mse': 'Function MSE',
            'function_mae': 'Function MAE',
            'function_rmse': 'Function RMSE',
            'function_r_squared': 'Function R²',
            'function_max_error': 'Function Max Error',
            'slope_error': 'Tangent Slope Error',
            'intercept_error': 'Tangent Intercept Error',
            'tangent_line_mse': 'Tangent Line MSE',
            'derivative_mse': 'Derivative MSE',
            'derivative_mae': 'Derivative MAE',
            'derivative_rmse': 'Derivative RMSE',
            'derivative_r_squared': 'Derivative R²',
            'derivative_max_error': 'Derivative Max Error',
        }
        
        rows = []
        for key, value in metrics.items():
            name = name_map.get(key, key.replace('_', ' ').title())
            if isinstance(value, float):
                if abs(value) < 0.001 or abs(value) > 10000:
                    formatted = f"{value:.6e}"
                else:
                    formatted = f"{value:.6f}"
            else:
                formatted = str(value)
            rows.append([name, formatted])
        
        table = tabulate(
            rows,
            headers=["Metric", "Value"],
            tablefmt="grid",
            colalign=("left", "right")
        )
        
        header = f"\n{'='*50}\n  {title}\n{'='*50}\n"
        return header + table
    
    @staticmethod
    def format_comparison_table(
        true_tangent: Dict,
        ai_tangent: Dict,
        comparison: Dict
    ) -> str:
        """
        Format a tangent line comparison table.
        
        Args:
            true_tangent: True tangent dictionary.
            ai_tangent: AI tangent dictionary.
            comparison: Comparison metrics dictionary.
        
        Returns:
            Formatted comparison table string.
        """
        rows = [
            ["Slope", f"{true_tangent['slope']:.6f}", f"{ai_tangent['slope']:.6f}",
             f"{comparison['slope_error']:.6e}"],
            ["Intercept", f"{true_tangent['intercept']:.6f}", f"{ai_tangent['intercept']:.6f}",
             f"{comparison['intercept_error']:.6e}"],
            ["f(x0)", f"{true_tangent['f_x0']:.6f}", f"{ai_tangent['f_x0']:.6f}",
             f"{comparison.get('f_x0_error', abs(true_tangent['f_x0'] - ai_tangent['f_x0'])):.6e}"],
        ]
        
        table = tabulate(
            rows,
            headers=["Property", "True Value", "AI Value", "Error"],
            tablefmt="grid",
            colalign=("left", "right", "right", "right")
        )
        
        header = "\n" + "=" * 60 + "\n  Tangent Line Comparison\n" + "=" * 60 + "\n"
        footer = f"\nTrue Equation:  {true_tangent['equation_point_form']}\n"
        footer += f"AI Equation:    {ai_tangent['equation_point_form']}\n"
        
        return header + table + footer
    
    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------
    
    @staticmethod
    def save_metrics(metrics: Dict[str, Any], filepath: str) -> str:
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Metrics dictionary.
            filepath: Output file path.
        
        Returns:
            Path to the saved file.
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        serializable = {}
        for key, value in metrics.items():
            if isinstance(value, (np.floating, np.integer)):
                serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            else:
                serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")
        return filepath
    
    @staticmethod
    def load_metrics(filepath: str) -> Dict[str, Any]:
        """
        Load metrics from a JSON file.
        
        Args:
            filepath: Path to the metrics JSON file.
        
        Returns:
            Metrics dictionary.
        
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metrics file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        
        logger.info(f"Metrics loaded from {filepath}")
        return metrics
