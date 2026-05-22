# =============================================================
# Tangent Line Computation Module
# AI-Based Neural Network System for Function Approximation
# Author: Mujeeb-Ur-Rehman Sahito (25-BSCS-43)
# =============================================================

"""
Tangent line computation for explicit and implicit functions.
Computes true analytic and AI-derived tangent lines, then compares them.

Tangent line equation: y = f'(x0)(x - x0) + f(x0)
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


# =============================================================
# Explicit Tangent Line Calculator
# =============================================================

class TangentLineCalculator:
    """
    Computes tangent lines for explicit functions y = f(x).
    
    Supports both true analytic tangent lines (using known derivative)
    and AI-derived tangent lines (using neural network + autograd).
    
    Tangent equation at (x0, f(x0)):
        y = f'(x0) * (x - x0) + f(x0)
    """
    
    @staticmethod
    def compute_tangent_line(
        x0: float,
        f_x0: float,
        f_prime_x0: float,
        x_range: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute the tangent line at point (x0, f(x0)) with slope f'(x0).
        
        Args:
            x0: The x-coordinate of the tangent point.
            f_x0: The function value at x0.
            f_prime_x0: The derivative value at x0.
            x_range: Optional array of x-values to evaluate the tangent line.
        
        Returns:
            Dictionary containing:
                - x0: tangent point x-coordinate
                - f_x0: function value at x0
                - slope: tangent slope (f'(x0))
                - intercept: y-intercept of the tangent line
                - equation: human-readable equation string
                - x_range: x-values (if provided)
                - y_tangent: tangent line y-values (if x_range provided)
        """
        slope = f_prime_x0
        intercept = f_x0 - slope * x0
        
        # Format equation string
        sign_intercept = "+" if intercept >= 0 else "-"
        equation = f"y = {slope:.4f}x {sign_intercept} {abs(intercept):.4f}"
        equation_point_form = f"y = {slope:.4f}(x - {x0:.4f}) + {f_x0:.4f}"
        
        result = {
            'x0': x0,
            'f_x0': f_x0,
            'slope': slope,
            'intercept': intercept,
            'equation': equation,
            'equation_point_form': equation_point_form,
            'x_range': x_range,
            'y_tangent': None
        }
        
        if x_range is not None:
            result['y_tangent'] = slope * (x_range - x0) + f_x0
        
        logger.info(f"Tangent line computed: {equation_point_form}")
        return result
    
    @staticmethod
    def compute_true_tangent(
        x0: float,
        func: Callable,
        derivative_func: Callable,
        x_range: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute the true analytic tangent line using known function and derivative.
        
        Args:
            x0: The x-coordinate of the tangent point.
            func: The analytic function f(x).
            derivative_func: The analytic derivative f'(x).
            x_range: Optional x-values for tangent line evaluation.
        
        Returns:
            Tangent line dictionary (see compute_tangent_line).
        """
        x0_arr = np.array([x0], dtype=np.float64)
        f_x0 = float(func(x0_arr)[0])
        f_prime_x0 = float(derivative_func(x0_arr)[0])
        
        result = TangentLineCalculator.compute_tangent_line(
            x0=x0,
            f_x0=f_x0,
            f_prime_x0=f_prime_x0,
            x_range=x_range
        )
        result['method'] = 'true_analytic'
        
        logger.info(
            f"True tangent at x0={x0:.4f}: "
            f"f(x0)={f_x0:.6f}, f'(x0)={f_prime_x0:.6f}"
        )
        return result
    
    @staticmethod
    def compute_ai_tangent(
        x0: float,
        model: torch.nn.Module,
        device: torch.device,
        x_range: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute the AI-derived tangent line using neural network + autograd.
        
        Uses automatic differentiation to compute the derivative of
        the neural network at x0, then constructs the tangent line.
        
        Args:
            x0: The x-coordinate of the tangent point.
            model: Trained neural network model.
            device: Compute device.
            x_range: Optional x-values for tangent line evaluation.
        
        Returns:
            Tangent line dictionary (see compute_tangent_line).
        """
        model.eval()
        
        # Create tensor with gradient tracking
        x_tensor = torch.FloatTensor([[x0]]).to(device)
        x_tensor.requires_grad_(True)
        
        # Forward pass
        y_pred = model(x_tensor)
        
        # Compute gradient (derivative) using autograd
        dy_dx = torch.autograd.grad(
            outputs=y_pred,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=False,
            retain_graph=False
        )[0]
        
        f_x0 = float(y_pred.item())
        f_prime_x0 = float(dy_dx.item())
        
        result = TangentLineCalculator.compute_tangent_line(
            x0=x0,
            f_x0=f_x0,
            f_prime_x0=f_prime_x0,
            x_range=x_range
        )
        result['method'] = 'ai_autograd'
        
        logger.info(
            f"AI tangent at x0={x0:.4f}: "
            f"f(x0)={f_x0:.6f}, f'(x0)={f_prime_x0:.6f}"
        )
        return result
    
    @staticmethod
    def compare_tangent_lines(
        true_tangent: Dict,
        ai_tangent: Dict
    ) -> Dict:
        """
        Compare true analytic and AI-derived tangent lines.
        
        Args:
            true_tangent: True tangent line dictionary.
            ai_tangent: AI-derived tangent line dictionary.
        
        Returns:
            Dictionary with comparison metrics:
                - slope_error: absolute difference in slopes
                - intercept_error: absolute difference in intercepts
                - slope_relative_error: relative slope error (%)
                - f_x0_error: error in function value at x0
                - tangent_mse: MSE between tangent line y-values
                - tangent_mae: MAE between tangent line y-values
        """
        slope_error = abs(true_tangent['slope'] - ai_tangent['slope'])
        intercept_error = abs(true_tangent['intercept'] - ai_tangent['intercept'])
        f_x0_error = abs(true_tangent['f_x0'] - ai_tangent['f_x0'])
        
        # Relative slope error (avoid division by zero)
        if abs(true_tangent['slope']) > 1e-10:
            slope_rel_error = slope_error / abs(true_tangent['slope']) * 100.0
        else:
            slope_rel_error = slope_error * 100.0
        
        result = {
            'slope_error': slope_error,
            'intercept_error': intercept_error,
            'slope_relative_error_pct': slope_rel_error,
            'f_x0_error': f_x0_error,
            'true_slope': true_tangent['slope'],
            'ai_slope': ai_tangent['slope'],
            'true_intercept': true_tangent['intercept'],
            'ai_intercept': ai_tangent['intercept'],
            'true_equation': true_tangent['equation_point_form'],
            'ai_equation': ai_tangent['equation_point_form']
        }
        
        # Tangent line MSE/MAE if y-values are available
        if (true_tangent['y_tangent'] is not None and
                ai_tangent['y_tangent'] is not None):
            diff = true_tangent['y_tangent'] - ai_tangent['y_tangent']
            result['tangent_mse'] = float(np.mean(diff ** 2))
            result['tangent_mae'] = float(np.mean(np.abs(diff)))
        
        logger.info(
            f"Tangent comparison: slope_error={slope_error:.6f}, "
            f"intercept_error={intercept_error:.6f}"
        )
        return result


# =============================================================
# Implicit Tangent Line Calculator
# =============================================================

class ImplicitTangentCalculator:
    """
    Computes tangent lines for implicit functions.
    
    Specifically handles circles of the form x² + y² = r².
    Uses implicit differentiation: dy/dx = -x/y.
    """
    
    @staticmethod
    def circle_tangent(
        x0: float,
        y0: Optional[float] = None,
        radius: float = 5.0,
        x_range: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute the true tangent line to the circle at (x0, y0).
        
        For circle x² + y² = r²:
            dy/dx = -x/y (implicit differentiation)
        
        Args:
            x0: The x-coordinate on the circle.
            y0: The y-coordinate. Computed from x0 if not given (upper semicircle).
            radius: The circle radius (default: 5.0).
            x_range: Optional x-values for tangent line evaluation.
        
        Returns:
            Tangent line dictionary with slope, intercept, equation, etc.
        
        Raises:
            ValueError: If (x0, y0) is not on the circle or y0 == 0.
        """
        # Compute y0 from upper semicircle if not provided
        if y0 is None:
            val = radius ** 2 - x0 ** 2
            if val < 0:
                raise ValueError(f"x0={x0} is outside the circle (|x0| > r={radius})")
            y0 = float(np.sqrt(val))
            logger.info(f"Computed y0={y0:.6f} from upper semicircle for x0={x0}")
        
        # Verify point is on the circle (with tolerance)
        residual = abs(x0 ** 2 + y0 ** 2 - radius ** 2)
        if residual > 0.01:
            logger.warning(
                f"Point ({x0}, {y0}) deviates from circle: "
                f"x²+y²={x0**2 + y0**2:.4f}, r²={radius**2}"
            )
        
        if abs(y0) < 1e-10:
            raise ValueError(f"Cannot compute tangent at y0={y0} (vertical tangent)")
        
        # Implicit derivative: dy/dx = -x/y
        slope = -x0 / y0
        intercept = y0 - slope * x0
        
        # Equation strings
        sign_intercept = "+" if intercept >= 0 else "-"
        equation = f"y = {slope:.4f}x {sign_intercept} {abs(intercept):.4f}"
        equation_point_form = f"y = {slope:.4f}(x - {x0:.4f}) + {y0:.4f}"
        
        result = {
            'x0': x0,
            'y0': y0,
            'f_x0': y0,
            'slope': slope,
            'intercept': intercept,
            'equation': equation,
            'equation_point_form': equation_point_form,
            'radius': radius,
            'method': 'implicit_differentiation',
            'x_range': x_range,
            'y_tangent': None
        }
        
        if x_range is not None:
            result['y_tangent'] = slope * (x_range - x0) + y0
        
        logger.info(
            f"Circle tangent at ({x0:.4f}, {y0:.4f}): "
            f"slope={slope:.6f}, {equation_point_form}"
        )
        return result
    
    @staticmethod
    def compute_ai_circle_tangent(
        x0: float,
        model: torch.nn.Module,
        device: torch.device,
        radius: float = 5.0,
        x_range: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute tangent to the circle using a neural network + autograd.
        
        The neural network maps x → y (upper semicircle), and autograd
        is used to compute dy/dx at the point x0.
        
        Args:
            x0: The x-coordinate on the circle.
            model: Trained circle approximator model.
            device: Compute device.
            radius: Circle radius.
            x_range: Optional x-values for tangent evaluation.
        
        Returns:
            Tangent line dictionary.
        """
        model.eval()
        
        # Forward pass with gradient tracking
        x_tensor = torch.FloatTensor([[x0]]).to(device)
        x_tensor.requires_grad_(True)
        
        y_pred = model(x_tensor)
        
        # Autograd derivative
        dy_dx = torch.autograd.grad(
            outputs=y_pred,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=False,
            retain_graph=False
        )[0]
        
        y0_pred = float(y_pred.item())
        slope = float(dy_dx.item())
        intercept = y0_pred - slope * x0
        
        sign_intercept = "+" if intercept >= 0 else "-"
        equation = f"y = {slope:.4f}x {sign_intercept} {abs(intercept):.4f}"
        equation_point_form = f"y = {slope:.4f}(x - {x0:.4f}) + {y0_pred:.4f}"
        
        result = {
            'x0': x0,
            'y0': y0_pred,
            'f_x0': y0_pred,
            'slope': slope,
            'intercept': intercept,
            'equation': equation,
            'equation_point_form': equation_point_form,
            'radius': radius,
            'method': 'ai_circle_autograd',
            'x_range': x_range,
            'y_tangent': None
        }
        
        if x_range is not None:
            result['y_tangent'] = slope * (x_range - x0) + y0_pred
        
        logger.info(
            f"AI circle tangent at x0={x0:.4f}: "
            f"y0={y0_pred:.6f}, slope={slope:.6f}"
        )
        return result
    
    @staticmethod
    def compare_circle_tangents(
        true_tangent: Dict,
        ai_tangent: Dict
    ) -> Dict:
        """
        Compare true implicit and AI-derived circle tangent lines.
        
        Args:
            true_tangent: True tangent from implicit differentiation.
            ai_tangent: AI tangent from neural network.
        
        Returns:
            Comparison metrics dictionary.
        """
        slope_error = abs(true_tangent['slope'] - ai_tangent['slope'])
        intercept_error = abs(true_tangent['intercept'] - ai_tangent['intercept'])
        y0_error = abs(true_tangent['y0'] - ai_tangent['y0'])
        
        result = {
            'slope_error': slope_error,
            'intercept_error': intercept_error,
            'y0_error': y0_error,
            'true_slope': true_tangent['slope'],
            'ai_slope': ai_tangent['slope'],
            'true_y0': true_tangent['y0'],
            'ai_y0': ai_tangent['y0'],
            'true_equation': true_tangent['equation_point_form'],
            'ai_equation': ai_tangent['equation_point_form']
        }
        
        if (true_tangent['y_tangent'] is not None and
                ai_tangent['y_tangent'] is not None):
            diff = true_tangent['y_tangent'] - ai_tangent['y_tangent']
            result['tangent_mse'] = float(np.mean(diff ** 2))
            result['tangent_mae'] = float(np.mean(np.abs(diff)))
        
        logger.info(
            f"Circle tangent comparison: slope_error={slope_error:.6f}, "
            f"y0_error={y0_error:.6f}"
        )
        return result
